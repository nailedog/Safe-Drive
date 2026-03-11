# SafeDrive Solution 2 — Техническая документация

## Обзор проекта

**SafeDrive** — система обнаружения засыпания водителя, работающая полностью на устройстве (on-device), без Wi-Fi, без облака, без задержки сети. Единственные входные данные — изображение с камеры. Единственный выход — сигнал тревоги на GPIO.

**Метод детекции:** Eye Aspect Ratio (EAR) — математический показатель степени открытия глаза, вычисляемый по координатам 12 ключевых точек лица (по 6 на каждый глаз) из 468 точек, которые возвращает нейронная сеть MediaPipe Face Landmark.

---

## Аппаратура

| Компонент | Характеристика |
|---|---|
| Плата | ESP32-S3 Eye |
| CPU | Xtensa LX7 Dual-Core, 240 MHz |
| Камера | OV5640 |
| Внутренний SRAM | 512 KB |
| PSRAM | 8 MB OPI (встроенный в чип) |
| Flash | 8 MB |
| USB | Native USB (CDC ACM), GPIO 19/20 |
| Сигнальный GPIO | GPIO 21 (ALARM_PIN) |
| LED GPIO | GPIO 2 (LIGHT_PIN, настраивается) |

---

## Алгоритм работы (Pipeline)

```
Камера OV5640
    ↓  QVGA 320×240, RGB565, 2 буфера в PSRAM
rgb565_to_rgb888()
    ↓  320×240×3 = 230 KB в PSRAM
crop_resize_nn()
    ↓  центральный квадрат 240×240 → ресайз nearest-neighbor → 192×192×3 = 110 KB
TFLite MicroInterpreter (INT8)
    ↓  MediaPipe Face Landmark, вход UINT8 [1,192,192,3]
468 точек лица (x,y,z нормированные [0..1]), выход FLOAT32 [1,1,1,1404]
    ↓
compute_ear() × 2 (левый и правый глаз)
    ↓  EAR = (dist(P1,P5) + dist(P2,P4)) / (2 × dist(P0,P3))
Усреднение: EAR = (left_ear + right_ear) / 2
    ↓
EAR < 0.22 дольше 2 секунд?
    ↓ Да               ↓ Нет
ТРЕВОГА (GPIO HIGH)   Нормально (GPIO LOW)
```

### Eye Aspect Ratio (EAR)

EAR — геометрический показатель, предложенный Сокой и Мотасом (2016). Для каждого глаза берутся 6 точек:
- P0, P3 — горизонтальные (уголки глаза)
- P1, P5 и P2, P4 — вертикальные пары

```
        P1   P2
   P0 •       • P3
        P5   P4

EAR = (|P1-P5| + |P2-P4|) / (2 × |P0-P3|)
```

При открытом глазе EAR ≈ 0.25–0.35. При закрытом — падает к 0.05–0.15. Порог 0.22 выбран экспериментально.

**Индексы точек MediaPipe Face Mesh (468 точек):**
- Левый глаз: 362, 385, 387, 263, 373, 380
- Правый глаз: 33, 160, 158, 133, 153, 144

---

## Нейронная сеть

### Исходная модель

**MediaPipe Face Landmark** (Google) — сверточная сеть для определения 468 трёхмерных точек лица.

| Параметр | Значение |
|---|---|
| Исходный файл | `face_landmark.tflite` (от Google MediaPipe) |
| Размер | 1,242,398 байт |
| Тип весов | FLOAT16 (для экономии места) |
| Активации | FLOAT32 |
| Вход | [1, 192, 192, 3] FLOAT32 |
| Выход 0 | [1, 1, 1, 1404] FLOAT32 — 468 точек × (x, y, z) |
| Выход 1 | [1, 1, 1, 1] FLOAT32 — вероятность наличия лица |
| Операции | CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, ADD, PRELU, DEQUANTIZE (×113) |

Оригинальная модель содержала 113 операций DEQUANTIZE: она хранила веса в FLOAT16, а перед каждым слоем конвертировала их в FLOAT32 на лету. Это было неэффективно.

### Этап 1: Graph Surgery (FLOAT16 → FLOAT32)

**Инструмент:** `quantize.py` (Python)
**Файл:** `/Users/gen/quantize_fl/quantize.py`

Модель в формате FlatBuffer была декодирована через `flatc` в JSON, после чего был выполнен прямой обход графа вычислений:

1. Все буферы с весами типа FLOAT16 перепакованы в FLOAT32:
   ```python
   f16_vals = struct.unpack(f'{n}e', data_bytes)   # читаем float16
   f32_data = struct.pack(f'{n}f', *f16_vals)       # пишем float32
   ```
2. Все 113 операций DEQUANTIZE удалены из графа.
3. Входные тензоры слоёв перемонтированы напрямую на FLOAT32-буферы.
4. Граф перекомпилирован обратно в `.tflite` через `flatc`.

**Результат:** `face_landmark_f32.tflite` — 2,447,776 байт, чистый FLOAT32, без DEQUANTIZE.

### Этап 2: INT8 квантизация

**Инструменты:**
- `tflite2tensorflow` — конвертация `.tflite` → `SavedModel` TensorFlow
- `TFLiteConverter` (TensorFlow 2.16.2) — квантизация через representative dataset
- `convert_int8.py` — скрипт квантизации

**Файлы:**
- `/Users/gen/quantize_fl/run_tflite2tf.py` — запуск tflite2tensorflow с патчем TF 2.16.2
- `/Users/gen/quantize_fl/convert_int8.py` — конвертация SavedModel → INT8 TFLite

**Процесс:**

```
face_landmark_f32.tflite
    ↓  tflite2tensorflow (конвертация в TF SavedModel)
savedmodel/
    ↓  TFLiteConverter.from_saved_model()
       optimizations = [DEFAULT]
       representative_dataset = 100 изображений лиц (calib_faces.npy)
       supported_ops = [TFLITE_BUILTINS_INT8, TFLITE_BUILTINS]
       inference_input_type  = tf.uint8
       inference_output_type = tf.float32
face_landmark_int8.tflite
```

**Технические сложности при квантизации:**

| Проблема | Решение |
|---|---|
| `tflite2tensorflow` использует путь `f'./{model}.json'` — не работает с абсолютным путём | Запуск `os.chdir(DIR)` + относительный путь к модели |
| TF 2.16.2 изменил сигнатуру `_get_tensor_details(tensor_index)` → `(tensor_index, subgraph_index)` | Monkey-patch с дефолтным параметром `subgraph_index=0` |
| Прямая правка FlatBuffer для квантизации не работает — `CONV_2D: output->type != input_type` | Использование полного пайплайна через SavedModel |

**Результат INT8 модели:**

| Параметр | Значение |
|---|---|
| Файл | `face_landmark_int8.tflite` |
| Размер | 759,256 байт (в 1.6× меньше исходного) |
| Вход | [1, 192, 192, 3] UINT8 (scale=1/255, zero_point=0) |
| Выход 0 | [1, 1, 1, 1404] FLOAT32 |
| Выход 1 | [1, 1, 1, 1] FLOAT32 |
| Операции | QUANTIZE(1), CONV_2D(25), PRELU(23), DEPTHWISE_CONV_2D(20), ADD(20), MAX_POOL_2D(5), PAD(3), DEQUANTIZE(2) |
| Веса | INT8 (4× меньше FLOAT32) |
| Активации | INT8 (4× меньше FLOAT32) |

### Преобразование модели в заголовочный файл C++

```bash
xxd -i face_landmark_int8.tflite > face_landmark_model.h
```

Результат — массив `const unsigned char g_face_landmark_model[]` (4.4 MB заголовочный файл), прошитый напрямую во Flash ESP32-S3.

---

## Оптимизации

### 1. INT8 квантизация (главная оптимизация)

| Метрика | FLOAT32 | INT8 | Улучшение |
|---|---|---|---|
| Время инференса | ~21 сек/кадр | ~1.6 сек/кадр | **13× быстрее** |
| Арена (TFLite) | ~4 MB | 640 KB | **6.4× меньше** |
| Размер модели (веса) | 2.4 MB | 759 KB | **3.2× меньше** |

Цифры из реального запуска на ESP32-S3 @ 240 MHz.

ESP32-S3 имеет инструкции SIMD для INT8 (через PIE extension), что объясняет непропорционально большое ускорение.

### 2. Отказ от BlazeFace (детектор лица)

Первоначальный пайплайн включал две нейронные сети:
- BlazeFace (детектор, 229 KB) — находит bbox лица
- Face Landmark (основная) — строит 468 точек

BlazeFace был убран. Вместо него — геометрический кроп центрального квадрата QVGA-кадра. Это:
- Экономит ~2 MB PSRAM (арена BlazeFace)
- Ускоряет пайплайн на ~30%
- Оправдано для водителя: лицо всегда в центре кадра

### 3. Буферы в PSRAM (ps_malloc)

Все крупные буферы выделяются в PSRAM, а не во внутреннем SRAM:
- Арена TFLite (2 MB) → `ps_malloc`
- RGB888 буфер (230 KB) → `ps_malloc`
- Crop-буфер (110 KB) → `ps_malloc`
- Camera framebuffer (×2, ~154 KB) → `CAMERA_FB_IN_PSRAM`

Внутренний SRAM (512 KB) остаётся почти свободным — используется только для стека, кода и небольших структур.

### 4. CAMERA_GRAB_LATEST

Режим захвата `CAMERA_GRAB_LATEST` с двойной буферизацией (`fb_count=2`) гарантирует, что после завершения инференса сразу доступен **свежий** кадр, без ожидания следующего кадра от сенсора.

### 5. Nearest-Neighbor ресайз (без float-вычислений)

Функция `crop_resize_nn()` использует целочисленную арифметику, без промежуточных FLOAT-массивов. Ресайз 240×240 → 192×192 выполняется in-place за один проход.

### 6. Прямое копирование входа (memcpy)

Для INT8 модели вход в формате UINT8 [0..255] совпадает с форматом пикселей из камеры. Копирование выполняется через `memcpy` — нет loop с делением на 255.0f.

### 7. Избирательное включение операций TFLite

Файл `trained_model_ops_define.h` содержит `#define EI_TFLITE_DISABLE_*` флаги — позволяет исключить из сборки неиспользуемые операции и типы данных. Для INT8 Face Landmark включены только нужные:
- CONV_2D F32 (для FLOAT32 fallback — закомментированы `_DISABLE_`)
- MAX_POOL_2D I8 — разкомментированы (были закрыты, что ломало INT8)
- DEPTHWISE_CONV_2D F32, STRIDED_SLICE F32

---

## Используемые технологии и библиотеки

| Технология | Версия | Роль |
|---|---|---|
| **Arduino IDE / arduino-cli** | 1.4.0 | Сборка и прошивка |
| **esp32 Arduino core** (Espressif) | 3.3.5 | HAL для ESP32-S3 |
| **esp32-camera** | встроена в core | Драйвер OV5640, LEDC XCLK |
| **Safe-Drive_inferencing** | 1.0.1 | Edge Impulse SDK: TFLite Micro runtime |
| **TFLite Micro** | через EI SDK | Инференс нейросети на MCU |
| **TensorFlow** | 2.16.2 | Квантизация модели (на ПК) |
| **tflite2tensorflow** | 1.22.0 | Конвертация .tflite → SavedModel (на ПК) |
| **flatc** | 25.12.19 | FlatBuffer компилятор (на ПК) |
| **Python** | 3.10 | Скрипты квантизации |

**Язык прошивки:** C++17 (Arduino framework)

**Язык скриптов квантизации:** Python 3.10

---

## Использование ресурсов

### Flash (партиция приложения, 3 MB Huge APP)

```
Всего:     3,145,728 байт (3 MB)
Занято:    1,979,643 байт (62%)
  ├─ Модель (Flash):   ~759 KB  (face_landmark_int8, встроена в .text)
  ├─ EI SDK + TFLite:  ~800 KB  (скомпилированный runtime)
  ├─ esp-camera + HAL: ~200 KB
  └─ Код приложения:    ~30 KB
Свободно:  1,166,085 байт (38%)
```

### Internal SRAM (512 KB)

```
Всего:    327,680 байт (из 512 KB — остальное резервирует ROM/BT)
Занято:    47,004 байт (14%)  — статические переменные, стек
Свободно: 280,676 байт (86%)
```

### PSRAM OPI (8 MB) — при работе

```
Всего:    8,192 KB
Занято (аллоцировано):
  ├─ TFLite arena:      2,048 KB (выделено) / 640 KB (реально используется)
  ├─ RGB888 буфер:        230 KB (320×240×3)
  ├─ Crop буфер:          110 KB (192×192×3)
  └─ Camera FB ×2:        307 KB (2 × 320×240×2, RGB565)
Итого занято:          ~2,695 KB
Свободно:              ~5,497 KB (67%)
```

### Производительность

| Метрика | Значение |
|---|---|
| Время инференса | ~1.6 сек/кадр |
| Кадровая частота | ~0.6 FPS |
| Время AllocateTensors | ~100 мс |
| Арена (реально) | 640,564 байт (625 KB) |
| CPU нагрузка | 100% Core 0 во время Invoke() |

---

## Файлы проекта

### Обязательные файлы для сборки

```
SafeDrive_Solution2_EAR/
├── SafeDrive_Solution2_EAR.ino   # Основной скетч
├── board_config.h                # Выбор модели платы (CAMERA_MODEL_ESP32S3_EYE)
├── camera_pins.h                 # GPIO-пины камеры для всех поддерживаемых плат
└── face_landmark_model.h         # INT8 модель в виде C-массива (4.4 MB)
```

### Библиотека (обязательна)

```
~/Documents/Arduino/libraries/
└── Safe-Drive_inferencing/       # Edge Impulse SDK v1.0.1
    └── src/
        ├── edge-impulse-sdk/
        │   └── tensorflow/lite/micro/   # TFLite Micro runtime
        └── tflite-model/
            └── trained_model_ops_define.h  # ← ИЗМЕНЁН (MAX_POOL_2D I8 включён)
```

> **Важно:** файл `trained_model_ops_define.h` был изменён вручную. При переустановке библиотеки изменения нужно восстановить (см. раздел ниже).

### Файлы квантизации (только для пересоздания модели)

```
~/quantize_fl/
├── face_landmark.tflite          # Исходная модель Google MediaPipe (FLOAT16)
├── face_landmark_f32.tflite      # После graph surgery (чистый FLOAT32)
├── face_landmark_int8.tflite     # Финальная INT8 модель (759 KB)
├── quantize.py                   # Graph surgery: FLOAT16 → FLOAT32
├── run_tflite2tf.py              # tflite → SavedModel (с патчем TF 2.16.2)
└── convert_int8.py               # SavedModel → INT8 TFLite
```

---

## Как собрать проект

### Требования

- Arduino IDE 2.x **или** arduino-cli ≥ 1.0
- esp32 Arduino core версии **3.3.5** (Espressif)
- Библиотека **Safe-Drive_inferencing v1.0.1**
- Плата ESP32-S3 Eye с PSRAM 8 MB

### Установка через arduino-cli

```bash
# 1. Установить core (если не установлен)
arduino-cli core install esp32:esp32@3.3.5

# 2. Скопировать библиотеку
cp -r Safe-Drive_inferencing ~/Documents/Arduino/libraries/

# 3. Применить патч к ops_define (после установки библиотеки)
#    В файле .../Safe-Drive_inferencing/src/tflite-model/trained_model_ops_define.h
#    закомментировать строки:
#      //#define EI_TFLITE_DISABLE_MAX_POOL_2D_IN_I8   1
#      //#define EI_TFLITE_DISABLE_MAX_POOL_2D_OUT_I8  1
```

### Настройки платы (Arduino IDE)

| Параметр | Значение |
|---|---|
| Board | ESP32S3 Dev Module |
| PSRAM | OPI PSRAM |
| Flash Size | 8MB |
| Partition Scheme | **Huge APP (3MB No OTA)** — обязательно! |
| CPU Frequency | 240MHz |
| USB CDC On Boot | **Enabled** |
| Upload Speed | 921600 |

### Сборка и прошивка (arduino-cli)

```bash
# Компиляция
arduino-cli compile \
  --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=8M,PartitionScheme=huge_app,PSRAM=opi,CPUFreq=240 \
  /path/to/SafeDrive_Solution2_EAR

# Прошивка
arduino-cli upload \
  --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=8M,PartitionScheme=huge_app,PSRAM=opi,CPUFreq=240 \
  -p /dev/cu.usbmodem1101 \
  /path/to/SafeDrive_Solution2_EAR
```

### После прошивки: порядок запуска

> **Важно для ESP32-S3 с CDCOnBoot=cdc:** после прошивки esptool выполняет "Hard resetting via RTS pin", что переводит плату в режим загрузчика. Прошивка не запустится автоматически.

**Для первого запуска после прошивки необходимо физически перезагрузить плату:**
- Отключить USB-кабель и подключить снова
- **ИЛИ** нажать кнопку RST на плате

### Настройка LED-пина

В `SafeDrive_Solution2_EAR.ino` найди строку:

```cpp
#define LIGHT_PIN 2
```

Замени `2` на GPIO пин своего LED. Убедись что пин не используется камерой или PSRAM (не используй GPIO 26–37 на ESP32-S3 с OPI PSRAM).

### Настройка порогов детекции

В начале `.ino` файла:

```cpp
static constexpr float EAR_THRESHOLD  = 0.22f;  // ниже — глаз закрыт
static constexpr float DROWSY_SECONDS = 2.0f;   // секунд до тревоги
```

---

## Пересоздание модели (если нужно)

Если нужно пересобрать `face_landmark_int8.tflite` с нуля:

```bash
cd ~/quantize_fl

# Шаг 1: graph surgery (FLOAT16 → FLOAT32)
python3.10 quantize.py
# → face_landmark_f32.tflite (2.4 MB)

# Шаг 2: конвертация в SavedModel
python3.10 run_tflite2tf.py
# → savedmodel/

# Шаг 3: INT8 квантизация
python3.10 convert_int8.py
# → face_landmark_int8.tflite (759 KB)

# Шаг 4: конвертация в C-заголовок
xxd -i face_landmark_int8.tflite > ../Documents/Arduino/SafeDrive_Solution2_EAR/face_landmark_model.h
# Вручную переименовать массив на g_face_landmark_model если нужно
```

**Требования Python-окружения для квантизации:**
```
tensorflow==2.16.2
tflite2tensorflow==1.22.0
flatc==25.12.19 (brew install flatbuffers)
numpy
```

---

## Диагностика (Serial Monitor, 115200 бод)

Нормальный вывод при старте:
```
=== SafeDrive Solution 2 — EAR TFLite On-Device ===
[OK] PSRAM свободно: 8189 KB
[OK] Арена: 2 MB выделено, свободно PSRAM: 6077 KB
[INFO] AllocateTensors вернул: 0 (0=OK, 1=ERROR)
[INFO] Использовано арены: 640564 bytes
[OK] FaceLandmark: вход [1,192,192,3], выход 1 значений, арена 625 KB
[OK] Камера: QVGA 320x240 RGB565
[OK] Готов! Начинаю детекцию...
```

В рабочем режиме:
```
[AWAKE]  EAR=0.271 (L=0.260 R=0.282)    ← глаза открыты
[DROWSY] EAR=0.174 (L=0.162 R=0.186)  1.6/2s  ← глаза закрыты, таймер
*** ВОДИТЕЛЬ ЗАСЫПАЕТ! ТРЕВОГА! ***     ← сработала тревога
[ALARM OFF]                              ← глаза открылись
```

| Сообщение | Значение |
|---|---|
| `[ERROR] PSRAM не найден!` | Неверные настройки платы — проверь PSRAM=OPI |
| `[ERROR] ps_malloc failed` | Недостаточно свободного PSRAM |
| `AllocateTensors вернул: 1` | Арена слишком мала или несовместимые ops |
| `[WARN] Invoke failed` | Ошибка инференса (обычно первый кадр) |

---

## Ограничения

- Нет детектора лица (BlazeFace удалён) — работает только когда лицо в центре кадра
- ~1.6 FPS — достаточно для обнаружения засыпания (реакция 2 сек), но не для плавного видео
- Нет OTA обновлений (партиция Huge APP без OTA-слота)
- LED-пин нужно настроить вручную под конкретную плату
