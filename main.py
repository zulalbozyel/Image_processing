import cv2
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import time

# ══════════════════════════════════════════
#  BURAYA KAMERA NUMARINI YAZ (test_camera.py ile bulduğun)
CAMERA_INDEX = 0
#  DİL SEÇ: "tr"=Türkçe, "es"=İspanyolca, "fr"=Fransızca
TARGET_LANGUAGE = "tr"
#  ALGILAMA HASSASIYETI (0.0 - 1.0 arası)
CONFIDENCE = 0.5
# ══════════════════════════════════════════

print("Model yükleniyor...")
model = YOLO("yolov8n.pt")   # İlk çalıştırmada otomatik indirilir (~6MB)
print("Model hazır!")

# Çeviri önbelleği (aynı kelimeyi tekrar çevirmemek için)
cache = {}

def translate(word):
    if word in cache:
        return cache[word]
    try:
        result = GoogleTranslator(source="en", target=TARGET_LANGUAGE).translate(word)
        cache[word] = result
        return result
    except Exception as e:
        return word  # İnternet yoksa İngilizce göster

# Kamerayı aç
print(f"Kamera {CAMERA_INDEX} açılıyor...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("HATA: Kamera açılamadı! CAMERA_INDEX numarasını değiştir.")
    exit()

print("Kamera açıldı! Çıkmak için 'q' tuşuna bas.")

# Kelime dosyasını aç
vocab = open("kelimeler.txt", "a", encoding="utf-8")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor...")
        break

    # YOLO ile nesne tespiti yap
    results = model(frame, conf=CONFIDENCE, verbose=False)

    for result in results:
        for box in result.boxes:
            # Koordinatları al
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            english_name = model.names[class_id]

            # Türkçeye çevir
            turkish_name = translate(english_name)

            # Etiketi hazırla
            label = f"{turkish_name}  ({english_name})"
            conf_text = f"%{int(confidence*100)}"

            # Yeşil kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Etiket arka planı
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x1, y1-35), (x1+tw+10, y1), (0, 200, 0), -1)

            # Etiket yazısı
            cv2.putText(frame, label, (x1+5, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            # Güven skoru
            cv2.putText(frame, conf_text, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # Kelimeyi dosyaya yaz
            vocab.write(f"{english_name} = {turkish_name}\n")
            vocab.flush()

    # Ekranda göster
    cv2.imshow("Dil Ogrenme - Nesne Algilama", frame)

    # Q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
vocab.close()
cv2.destroyAllWindows()
print("Program kapandı. Kelimelerin kelimeler.txt dosyasına kaydedildi!")
