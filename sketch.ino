/*
 * Automated Optical Inspection (AOI) - Edge-AI defect classifier
 * Board: ESP32 DevKit. Serial over USB @115200 (shows in Wokwi monitor).
 *
 * A tiny CNN (trained in TensorFlow, exported to model_weights.h) runs ENTIRELY
 * on-device and classifies 16x16 grayscale surface patches as PASS or DEFECT,
 * exactly like a production-line vision station. No host PC, no serial bridge.
 *
 * PIN MAP (must match diagram.json exactly):
 *   OLED SSD1306 I2C : SDA=GPIO21, SCL=GPIO22, VCC=3V3, GND=GND
 *   NEXT  button     : GPIO14 -> GND  (INPUT_PULLUP, pressed = LOW)
 *   AUTO  button     : GPIO27 -> GND  (INPUT_PULLUP, pressed = LOW)
 *   GREEN  "PASS"  LED : GPIO25 -> 220R -> LED -> GND
 *   RED    "DEFECT"LED : GPIO26 -> 220R -> LED -> GND
 *   YELLOW "BUSY"  LED : GPIO33 -> 220R -> LED -> GND
 *   BUZZER            : GPIO32 -> buzzer -> GND
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model_weights.h"
#include "test_images.h"

// ---------- Result type + forward declarations (Arduino .ino needs these) ----------
struct Result { int cls; float conf; float p[2]; };
static Result classify(int idx);
static void drawImage(int idx);
static void render(int idx, const Result& r);
static void runInspection(int idx);

// ---------- Pins ----------
static const int PIN_SDA    = 21;
static const int PIN_SCL    = 22;
static const int PIN_NEXT   = 14;
static const int PIN_AUTO   = 27;
static const int PIN_PASS   = 25;
static const int PIN_DEFECT = 26;
static const int PIN_BUSY   = 33;
static const int PIN_BUZZER = 32;

// ---------- OLED ----------
#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

// ---------- Feature-map buffers (static, no heap) ----------
static float fmIn[IMG_SIZE*IMG_SIZE*C0_IN];            // 16x16x1
static float fm0 [IMG_SIZE*IMG_SIZE*C0_OUT];           // 16x16x8
static float fm1 [(IMG_SIZE/2)*(IMG_SIZE/2)*C0_OUT];   // 8x8x8
static float fm2 [(IMG_SIZE/2)*(IMG_SIZE/2)*C1_OUT];   // 8x8x16
static float fm3 [(IMG_SIZE/4)*(IMG_SIZE/4)*C1_OUT];   // 4x4x16 = 256
static float d0  [D0_OUT];
static float d1  [D1_OUT];

// ---------- Layers: replicate Keras/numpy EXACTLY ----------
static void conv_same_relu(const float* in, int H, int W, int INc,
                           const float* wt, const float* bs, int OUTc,
                           float* out) {
  for (int oc = 0; oc < OUTc; oc++) {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        float acc = bs[oc];
        for (int kh = 0; kh < 3; kh++) {
          int ii = i + kh - 1;
          if (ii < 0 || ii >= H) continue;
          for (int kw = 0; kw < 3; kw++) {
            int jj = j + kw - 1;
            if (jj < 0 || jj >= W) continue;
            for (int ic = 0; ic < INc; ic++) {
              float v  = in[(ii*W + jj)*INc + ic];
              float wv = wt[((kh*3 + kw)*INc + ic)*OUTc + oc];
              acc += v * wv;
            }
          }
        }
        if (acc < 0.0f) acc = 0.0f;              // ReLU
        out[(i*W + j)*OUTc + oc] = acc;
      }
    }
  }
}

static void maxpool2(const float* in, int H, int W, int C, float* out) {
  int OH = H/2, OW = W/2;
  for (int c = 0; c < C; c++)
    for (int i = 0; i < OH; i++)
      for (int j = 0; j < OW; j++) {
        float m = -3.4e38f;
        for (int di = 0; di < 2; di++)
          for (int dj = 0; dj < 2; dj++) {
            float v = in[(((2*i+di)*W) + (2*j+dj))*C + c];
            if (v > m) m = v;
          }
        out[((i*OW) + j)*C + c] = m;
      }
}

static void dense(const float* in, int INlen, const float* wt, const float* bs,
                  int OUT, float* out, bool doRelu) {
  for (int o = 0; o < OUT; o++) {
    float acc = bs[o];
    for (int i = 0; i < INlen; i++)
      acc += in[i] * wt[i*OUT + o];
    if (doRelu && acc < 0.0f) acc = 0.0f;
    out[o] = acc;
  }
}

static Result classify(int idx) {
  for (int p = 0; p < IMG_PIXELS; p++)
    fmIn[p] = (float)TEST_IMAGES[idx][p] / 255.0f;

  conv_same_relu(fmIn, IMG_SIZE, IMG_SIZE, C0_IN, conv0_w, conv0_b, C0_OUT, fm0);
  maxpool2(fm0, IMG_SIZE, IMG_SIZE, C0_OUT, fm1);
  conv_same_relu(fm1, IMG_SIZE/2, IMG_SIZE/2, C1_IN, conv1_w, conv1_b, C1_OUT, fm2);
  maxpool2(fm2, IMG_SIZE/2, IMG_SIZE/2, C1_OUT, fm3);
  dense(fm3, FLAT_SIZE, dense0_w, dense0_b, D0_OUT, d0, true);
  dense(d0,  D0_OUT,    dense1_w, dense1_b, D1_OUT, d1, false);

  float mx = d1[0] > d1[1] ? d1[0] : d1[1];     // stable softmax
  float e0 = expf(d1[0]-mx), e1 = expf(d1[1]-mx);
  float s  = e0 + e1;
  Result r;
  r.p[0] = e0/s; r.p[1] = e1/s;
  r.cls  = (r.p[1] >= r.p[0]) ? 1 : 0;
  r.conf = r.p[r.cls];
  return r;
}

// ---------- State ----------
static int inspected = 0, defects = 0, curr = 0;
static bool autoMode = false;
static unsigned long busyOffAt = 0, buzzOffAt = 0;

// ---------- Display ----------
static void drawImage(int idx) {
  int mn = 255, mx = 0;
  for (int p = 0; p < IMG_PIXELS; p++) { int v = TEST_IMAGES[idx][p]; if (v<mn) mn=v; if (v>mx) mx=v; }
  int th = (mn + mx) / 2;
  const int ox = 0, oy = 12, s = 3;
  for (int y = 0; y < IMG_SIZE; y++)
    for (int x = 0; x < IMG_SIZE; x++) {
      int v = TEST_IMAGES[idx][y*IMG_SIZE + x];
      if (v <= th) display.fillRect(ox + x*s, oy + y*s, s, s, SSD1306_WHITE);
    }
  display.drawRect(ox-1, oy-1, IMG_SIZE*s + 2, IMG_SIZE*s + 2, SSD1306_WHITE);
}

static void render(int idx, const Result& r) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(TEST_NAMES[idx]);
  drawImage(idx);
  int tx = 56;
  display.setTextSize(2);
  display.setCursor(tx, 14);
  display.print(r.cls == 0 ? "PASS" : "DEF");
  display.setTextSize(1);
  display.setCursor(tx, 34);
  display.print("conf "); display.print((int)(r.conf*100.0f + 0.5f)); display.print("%");
  display.setCursor(tx, 48);
  display.print("I:"); display.print(inspected);
  display.print(" D:"); display.print(defects);
  display.display();
}

static void runInspection(int idx) {
  digitalWrite(PIN_BUSY, HIGH);
  Result r = classify(idx);
  busyOffAt = millis() + 120;

  inspected++;
  if (r.cls == 1) defects++;

  digitalWrite(PIN_PASS,   r.cls == 0 ? HIGH : LOW);
  digitalWrite(PIN_DEFECT, r.cls == 1 ? HIGH : LOW);
  if (r.cls == 1) { tone(PIN_BUZZER, 2500); buzzOffAt = millis() + 150; }

  render(idx, r);

  Serial.print("["); Serial.print(idx); Serial.print("] ");
  Serial.print(TEST_NAMES[idx]);
  Serial.print(" truth="); Serial.print(TEST_LABELS[idx] == 0 ? "PASS" : "DEFECT");
  Serial.print(" pred=");  Serial.print(r.cls == 0 ? "PASS" : "DEFECT");
  Serial.print(" conf=");  Serial.print(r.conf*100.0f, 1); Serial.print("%");
  Serial.print("  -> ");   Serial.println(r.cls == TEST_LABELS[idx] ? "OK" : "MISMATCH");
}

// ---------- Setup / Loop ----------
void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(PIN_NEXT, INPUT_PULLUP);
  pinMode(PIN_AUTO, INPUT_PULLUP);
  pinMode(PIN_PASS, OUTPUT);
  pinMode(PIN_DEFECT, OUTPUT);
  pinMode(PIN_BUSY, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_PASS, LOW); digitalWrite(PIN_DEFECT, LOW); digitalWrite(PIN_BUSY, LOW);

  Wire.begin(PIN_SDA, PIN_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR))
    Serial.println("SSD1306 init failed");
  display.clearDisplay(); display.display();

  Serial.println();
  Serial.println("=== Edge-AI AOI Defect Classifier (ESP32) ===");
  Serial.print("Input: "); Serial.print(IMG_SIZE); Serial.print("x"); Serial.print(IMG_SIZE);
  Serial.println(" grayscale");
  Serial.println("CNN: Conv8 - Pool - Conv16 - Pool - Dense16 - Dense2(softmax)");
  Serial.print("Embedded test parts: "); Serial.println(N_TEST);
  Serial.println("---------------------------------------------");

  runInspection(curr);
  autoMode = true;
}

void loop() {
  // ---- TEMP I2C scanner (prints once) ----
  static bool scanned = false;
  if (!scanned) {
    scanned = true;
    Serial.println("I2C scan:");
    for (byte a = 1; a < 127; a++) {
      Wire.beginTransmission(a);
      if (Wire.endTransmission() == 0) {
        Serial.print("  found device at 0x");
        Serial.println(a, HEX);
      }
    }
    Serial.println("scan done");
  }
  // ---- end scanner ----

  unsigned long now = millis();
  if (busyOffAt && now >= busyOffAt) { digitalWrite(PIN_BUSY, LOW); busyOffAt = 0; }
  if (buzzOffAt && now >= buzzOffAt) { noTone(PIN_BUZZER); buzzOffAt = 0; }

  static bool nPrev = HIGH; static unsigned long nT = 0;
  bool nNow = digitalRead(PIN_NEXT);
  if (nPrev == HIGH && nNow == LOW && now - nT > 180) {
    nT = now; curr = (curr + 1) % N_TEST; runInspection(curr);
  }
  nPrev = nNow;

  static bool aPrev = HIGH; static unsigned long aT = 0;
  bool aNow = digitalRead(PIN_AUTO);
  if (aPrev == HIGH && aNow == LOW && now - aT > 180) {
    aT = now; autoMode = !autoMode;
    Serial.print("AUTO mode: "); Serial.println(autoMode ? "ON" : "OFF");
  }
  aPrev = aNow;

  static unsigned long lastAuto = 0;
  if (autoMode && now - lastAuto > 2000) {
    lastAuto = now; curr = (curr + 1) % N_TEST; runInspection(curr);
  }
}