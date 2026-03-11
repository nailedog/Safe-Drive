# Загрузка TFLite моделей для Solution 2 EAR

## Нужные модели

| Модель | Файл | Размер | Назначение |
|---|---|---|---|
| BlazeFace short-range | `face_detection_short_range.tflite` | ~230 KB | Детекция лица |
| MediaPipe Face Landmark | `face_landmark.tflite` | ~1.7 MB | 468 точек лица |

---

## Шаг 1 — Скачать модели

### BlazeFace (детектор лица)
Скачать вручную с Google MediaPipe:
```
face_detection_short_range.tflite
```

### Face Landmark (468 точек)
```
face_landmark.tflite
```

Оба файла доступны в официальном MediaPipe репозитории на GitHub в папке `mediapipe/modules/face_detection/` и `mediapipe/modules/face_landmark/`.

---

## Шаг 2 — Конвертация в C-массивы

Выполнить в терминале из папки со скачанными файлами:

```bash
# Для Mac/Linux используется утилита xxd
xxd -i face_detection_short_range.tflite > face_detection_model.h
xxd -i face_landmark.tflite > face_landmark_model.h

# Скопировать оба .h файла в папку SafeDrive_Solution2_EAR/
cp face_detection_model.h /path/to/SafeDrive_Solution2_EAR/
cp face_landmark_model.h  /path/to/SafeDrive_Solution2_EAR/
```

---

## Шаг 3 — Исправить имена массивов в .h файлах

`xxd` генерирует имена на основе имени файла. Нужно переименовать:

**face_detection_model.h** — найти и заменить:
```c
// ДО (сгенерировано xxd):
unsigned char face_detection_short_range_tflite[] = { ... };
unsigned int  face_detection_short_range_tflite_len = ...;

// ПОСЛЕ (нужно переименовать в):
unsigned char g_face_detection_model[] = { ... };
unsigned int  g_face_detection_model_len = ...;
```

**face_landmark_model.h** — найти и заменить:
```c
// ДО:
unsigned char face_landmark_tflite[] = { ... };
unsigned int  face_landmark_tflite_len = ...;

// ПОСЛЕ:
unsigned char g_face_landmark_model[] = { ... };
unsigned int  g_face_landmark_model_len = ...;
```

---

## Шаг 4 — Установить библиотеку TFLite

В Arduino IDE → Manage Libraries → найти:
```
TensorFlowLite_ESP32
```
от автора **Limengdu0117** и установить.

---

## Шаг 5 — Настройки Arduino IDE

- **Board**: ESP32S3 Dev Module (или Seeed XIAO ESP32S3)
- **PSRAM**: OPI PSRAM (обязательно!)
- **Partition Scheme**: Huge APP (3MB No OTA / 1MB SPIFFS)  ← важно для 1.7MB модели
- **Flash Size**: 8MB
- **CPU Frequency**: 240MHz
- **Upload Speed**: 921600

---

## Структура файлов после подготовки

```
SafeDrive_Solution2_EAR/
├── SafeDrive_Solution2_EAR.ino   ← главный скетч
├── board_config.h                ← скопировать из CameraWebServer проекта
├── camera_pins.h                 ← скопировать из CameraWebServer проекта
├── face_detection_model.h        ← сгенерировать (BlazeFace)
└── face_landmark_model.h         ← сгенерировать (Face Landmark)
```
