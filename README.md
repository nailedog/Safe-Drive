# Safe Drive

Система обнаружения засыпания водителя на ESP32-S3. Работает полностью на устройстве — без Wi-Fi и облака. Анализирует положение глаз через камеру и подаёт сигнал тревоги если водитель засыпает.

---

## Требования

### Железо
- Плата: **ESP32-S3 Eye**
- Камера: OV5640 (встроена в плату)
- PSRAM: 8 MB OPI

### Программное обеспечение
- [Arduino IDE 2.x](https://www.arduino.cc/en/software) или arduino-cli ≥ 1.0
- **esp32 Arduino core версии 3.3.5** (Espressif)

---

## Установка

### 1. Скачать репозиторий
```bash
git clone https://github.com/nailedog/Safe-Drive.git
cd Safe-Drive
```

### 2. Установить esp32 core
В Arduino IDE: `File → Preferences → Additional boards manager URLs` добавить:
```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
Затем `Tools → Board → Boards Manager` → найти **esp32 by Espressif** → установить версию **3.3.5**.

Или через arduino-cli:
```bash
arduino-cli core install esp32:esp32@3.3.5
```

### 3. Установить библиотеку
Скопировать папку из репозитория в папку библиотек Arduino:
```bash
cp -r libraries/Safe-Drive_inferencing ~/Documents/Arduino/libraries/
```
> На Windows: `C:\Users\<username>\Documents\Arduino\libraries\`

### 4. Применить патч к библиотеке
Открыть файл:
```
~/Documents/Arduino/libraries/Safe-Drive_inferencing/src/tflite-model/trained_model_ops_define.h
```
Найти и закомментировать две строки (добавить `//` в начале):
```cpp
//#define EI_TFLITE_DISABLE_MAX_POOL_2D_IN_I8   1
//#define EI_TFLITE_DISABLE_MAX_POOL_2D_OUT_I8  1
```
> ⚠️ Без этого шага проект не скомпилируется корректно.

---

## Настройки платы в Arduino IDE

`Tools` меню — выставить следующие параметры:

| Параметр | Значение |
|---|---|
| Board | ESP32S3 Dev Module |
| PSRAM | **OPI PSRAM** |
| Flash Size | **8MB** |
| Partition Scheme | **Huge APP (3MB No OTA)** |
| CPU Frequency | 240MHz |
| USB CDC On Boot | **Enabled** |
| Upload Speed | 921600 |

---

## Компиляция и прошивка

### Через Arduino IDE
1. Открыть `Safe Drive/Safe Drive.ino`
2. Выставить настройки платы (см. выше)
3. Нажать **Upload**

### Через arduino-cli
```bash
# Компиляция
arduino-cli compile \
  --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=8M,PartitionScheme=huge_app,PSRAM=opi,CPUFreq=240 \
  "Safe Drive"

# Прошивка (заменить порт на свой)
arduino-cli upload \
  --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=8M,PartitionScheme=huge_app,PSRAM=opi,CPUFreq=240 \
  -p /dev/cu.usbmodem1101 \
  "Safe Drive"
```

---

## После прошивки

После загрузки прошивки **нажать кнопку RST на плате** или переподключить USB — плата не запускается автоматически при использовании CDC.

Открыть Serial Monitor (115200 бод). Нормальный вывод при старте:
```
=== SafeDrive Solution 2 — EAR TFLite On-Device ===
[OK] PSRAM свободно: 8189 KB
[OK] Арена: 2 MB выделено, свободно PSRAM: 6077 KB
[OK] FaceLandmark: вход [1,192,192,3], выход 1 значений, арена 625 KB
[OK] Камера: QVGA 320x240 RGB565
[OK] Готов! Начинаю детекцию...
```

---

## Настройка порогов

В файле `Safe Drive/Safe Drive.ino`:
```cpp
static constexpr float EAR_THRESHOLD  = 0.22f;  // порог закрытия глаза
static constexpr float DROWSY_SECONDS = 2.0f;   // секунд до тревоги
```
