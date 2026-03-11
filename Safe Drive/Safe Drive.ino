#include <Safe-Drive_inferencing.h>

// Перенаправляем ei_printf (TFLite Micro ошибки) в USB CDC Serial
void ei_printf(const char* fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Serial.print(buf);
}
#include "edge-impulse-sdk/tensorflow/lite/micro/all_ops_resolver.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/micro_interpreter.h"
#include "edge-impulse-sdk/tensorflow/lite/schema/schema_generated.h"

#include "esp_camera.h"
#include "board_config.h"
#include "face_landmark_model.h"   // g_face_landmark_model[], g_face_landmark_model_len

static constexpr float EAR_THRESHOLD  = 0.22f;
static constexpr float DROWSY_SECONDS = 2.0f;

#if defined(LED_GPIO_NUM) && LED_GPIO_NUM >= 0
  #define ALARM_PIN LED_GPIO_NUM
#else
  #define ALARM_PIN 21
#endif

#define LIGHT_PIN  42
#define BUZZER_PIN 41

static constexpr int BLINK_PERIOD_MS = 150;  // ~3.3 Гц мигание

static bool s_alarm = false;  // forward — используется в alarmTask

static void alarmTask(void*) {
  bool state = false;
  for (;;) {
    if (s_alarm) {
      state = !state;
      digitalWrite(LIGHT_PIN,  state ? HIGH : LOW);
      digitalWrite(BUZZER_PIN, state ? HIGH : LOW);
    } else {
      if (state) {
        digitalWrite(LIGHT_PIN,  LOW);
        digitalWrite(BUZZER_PIN, LOW);
        state = false;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(BLINK_PERIOD_MS));
  }
}

static constexpr int LANDMARK_W = 192;
static constexpr int LANDMARK_H = 192;
static constexpr int CAM_W      = 320;
static constexpr int CAM_H      = 240;

static constexpr int LEFT_EYE[6]  = {362, 385, 387, 263, 373, 380};
static constexpr int RIGHT_EYE[6] = { 33, 160, 158, 133, 153, 144};

static tflite::AllOpsResolver     s_resolver;
static const tflite::Model*       s_lm_model  = nullptr;
static tflite::MicroInterpreter*  s_lm_interp = nullptr;
static uint8_t* s_lm_arena = nullptr;
static constexpr int LM_ARENA = 2 * 1024 * 1024;  // 2 MB (INT8 activations are 4x smaller)

static uint8_t* s_rgb888  = nullptr;  // 320×240×3  = 230 KB
static uint8_t* s_lm_buf  = nullptr;  // 192×192×3  = 110 KB

static unsigned long s_closed_ms = 0;

static void rgb565_to_rgb888(const uint8_t* src, uint8_t* dst, int n) {
  for (int i = 0; i < n; i++) {
    uint16_t px = ((uint16_t)src[i*2] << 8) | src[i*2+1];
    dst[i*3]   = (uint8_t)((px >> 8) & 0xF8);
    dst[i*3+1] = (uint8_t)((px >> 3) & 0xFC);
    dst[i*3+2] = (uint8_t)((px << 3) & 0xF8);
  }
}

static void crop_resize_nn(const uint8_t* src, int src_w, int src_h,
                            int cx1, int cy1, int cx2, int cy2,
                            uint8_t* dst, int dst_w, int dst_h) {
  int cw = max(1, cx2 - cx1), ch = max(1, cy2 - cy1);
  for (int dy = 0; dy < dst_h; dy++) {
    int sy = cy1 + (int)((dy + 0.5f) * ch / dst_h);
    sy = max(0, min(sy, src_h - 1));
    for (int dx = 0; dx < dst_w; dx++) {
      int sx = cx1 + (int)((dx + 0.5f) * cw / dst_w);
      sx = max(0, min(sx, src_w - 1));
      const uint8_t* p = src + (sy * src_w + sx) * 3;
      uint8_t*       q = dst + (dy * dst_w + dx) * 3;
      q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
    }
  }
}

static float dist2d(float x1, float y1, float x2, float y2) {
  float dx = x1-x2, dy = y1-y2;
  return sqrtf(dx*dx + dy*dy);
}

static float compute_ear(const float* lm, const int* idx) {
  float px[6], py[6];
  for (int i = 0; i < 6; i++) {
    px[i] = lm[idx[i]*3+0] * LANDMARK_W;
    py[i] = lm[idx[i]*3+1] * LANDMARK_H;
  }
  float num = dist2d(px[1],py[1],px[5],py[5]) + dist2d(px[2],py[2],px[4],py[4]);
  float den = 2.0f * dist2d(px[0],py[0],px[3],py[3]);
  return (den > 0.5f) ? (num / den) : 0.0f;
}

static bool init_camera() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0; cfg.ledc_timer = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO_NUM; cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM; cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM; cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM; cfg.pin_d7 = Y9_GPIO_NUM;
  cfg.pin_xclk = XCLK_GPIO_NUM; cfg.pin_pclk  = PCLK_GPIO_NUM;
  cfg.pin_vsync = VSYNC_GPIO_NUM; cfg.pin_href  = HREF_GPIO_NUM;
  cfg.pin_sccb_sda = SIOD_GPIO_NUM; cfg.pin_sccb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn = PWDN_GPIO_NUM; cfg.pin_reset = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.frame_size   = FRAMESIZE_QVGA;
  cfg.pixel_format = PIXFORMAT_RGB565;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;
  cfg.fb_count     = 2;
  if (esp_camera_init(&cfg) != ESP_OK) return false;
  sensor_t* s = esp_camera_sensor_get();
  if (s->id.PID == OV5640_PID) {
    s->set_vflip(s, 1); s->set_hmirror(s, 1); s->set_brightness(s, 1);
  }
  return true;
}

static bool init_model() {
  s_lm_arena = (uint8_t*)ps_malloc(LM_ARENA);
  if (!s_lm_arena) {
    Serial.printf("[ERROR] ps_malloc(%d MB) провалился. Свободно: %u KB\n",
                  LM_ARENA/1024/1024, ESP.getFreePsram()/1024);
    return false;
  }
  Serial.printf("[OK] Арена: %d MB выделено, свободно PSRAM: %u KB\n",
                LM_ARENA/1024/1024, ESP.getFreePsram()/1024);

  s_lm_model = tflite::GetModel(g_face_landmark_model);
  if (s_lm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Версия схемы модели не совпадает");
    return false;
  }

  static tflite::MicroInterpreter interp(
      s_lm_model, s_resolver, s_lm_arena, LM_ARENA);
  s_lm_interp = &interp;

  Serial.println("[INFO] Запускаю AllocateTensors...");
  Serial.flush();
  delay(100);

  TfLiteStatus alloc_status = s_lm_interp->AllocateTensors(true);
  Serial.printf("[INFO] AllocateTensors вернул: %d (0=OK, 1=ERROR)\n", (int)alloc_status);
  Serial.printf("[INFO] Использовано арены: %u bytes\n", s_lm_interp->arena_used_bytes());
  Serial.printf("[INFO] Входов: %d, Выходов: %d\n",
                s_lm_interp->inputs_size(), s_lm_interp->outputs_size());
  Serial.flush();

  if (alloc_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors провалился: face landmark");
    return false;
  }

  auto* ti = s_lm_interp->input(0);
  auto* to = s_lm_interp->output(0);
  Serial.printf("[OK] FaceLandmark: вход [%d,%d,%d,%d], выход %d значений, арена %u KB\n",
                ti->dims->data[0], ti->dims->data[1],
                ti->dims->data[2], ti->dims->data[3],
                to->dims->data[1],
                s_lm_interp->arena_used_bytes() / 1024);
  return true;
}

static void set_alarm(bool on) {
  if (s_alarm == on) return;
  s_alarm = on;
  digitalWrite(ALARM_PIN, on ? HIGH : LOW);
  // LIGHT_PIN и BUZZER_PIN управляются из alarmTask
  Serial.println(on ? "\n*** ВОДИТЕЛЬ ЗАСЫПАЕТ! ТРЕВОГА! ***" : "[ALARM OFF]");
}

void setup() {
  pinMode(ALARM_PIN, OUTPUT);
  digitalWrite(ALARM_PIN, LOW);
  pinMode(LIGHT_PIN, OUTPUT);
  digitalWrite(LIGHT_PIN, LOW);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  xTaskCreate(alarmTask, "alarm", 1024, nullptr, 1, nullptr);

  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 5000) {}
  delay(300);
  Serial.println("\n=== SafeDrive Solution 2 — EAR TFLite On-Device ===");

  if (!psramFound()) {
    Serial.println("[ERROR] PSRAM не найден!"); while (true) delay(1000);
  }
  Serial.printf("[OK] PSRAM свободно: %u KB\n", ESP.getFreePsram() / 1024);

  if (!init_model()) {
    Serial.println("[ERROR] Модель не загружена"); while (true) delay(1000);
  }

  s_rgb888 = (uint8_t*)ps_malloc(CAM_W * CAM_H * 3);
  s_lm_buf = (uint8_t*)ps_malloc(LANDMARK_W * LANDMARK_H * 3);
  if (!s_rgb888 || !s_lm_buf) {
    Serial.println("[ERROR] Нет PSRAM для буферов"); while (true) delay(1000);
  }

  if (!init_camera()) {
    Serial.println("[ERROR] Камера не инициализирована"); while (true) delay(1000);
  }
  Serial.println("[OK] Камера: QVGA 320x240 RGB565");
  Serial.println("[OK] Готов! Начинаю детекцию...\n");
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { delay(30); return; }
  rgb565_to_rgb888(fb->buf, s_rgb888, CAM_W * CAM_H);
  esp_camera_fb_return(fb);

  int margin = (CAM_W - CAM_H) / 2;  
  crop_resize_nn(s_rgb888, CAM_W, CAM_H,
                 margin, 0, CAM_W - margin, CAM_H,
                 s_lm_buf, LANDMARK_W, LANDMARK_H);

  TfLiteTensor* lm_in = s_lm_interp->input(0);
  const int lm_n = LANDMARK_W * LANDMARK_H * 3;
  if (lm_in->type == kTfLiteUInt8) {
    memcpy(lm_in->data.uint8, s_lm_buf, lm_n); 
  } else {
    for (int i = 0; i < lm_n; i++)
      lm_in->data.f[i] = s_lm_buf[i] / 255.0f; 
  }

  if (s_lm_interp->Invoke() != kTfLiteOk) {
    Serial.println("[WARN] Invoke failed"); return;
  }

  const float* lm = s_lm_interp->output(0)->data.f;
  float left_ear  = compute_ear(lm, LEFT_EYE);
  float right_ear = compute_ear(lm, RIGHT_EYE);
  float ear       = (left_ear + right_ear) * 0.5f;
  bool  closed    = (ear < EAR_THRESHOLD);

  unsigned long now = millis();
  if (closed) {
    if (!s_closed_ms) s_closed_ms = now;
    float sec = (now - s_closed_ms) / 1000.0f;
    Serial.printf("[DROWSY] EAR=%.3f (L=%.3f R=%.3f)  %.1f/%.0fs\n",
                  ear, left_ear, right_ear, sec, DROWSY_SECONDS);
    if (sec >= DROWSY_SECONDS) set_alarm(true);
  } else {
    s_closed_ms = 0;
    set_alarm(false);
    Serial.printf("[AWAKE]  EAR=%.3f (L=%.3f R=%.3f)\n",
                  ear, left_ear, right_ear);
  }
}
