import cv2
from ultralytics import YOLO
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time

# ══════════════════════════════════════════
CAMERA_INDEX = 0          # iPhone veya Mac kamera indexin
TARGET_LANGUAGE = "tr"    # Hedef dil
CONFIDENCE = 0.35         # Algılama hassasiyeti düşürüldü (daha fazla nesne bulması için)
# ══════════════════════════════════════════

# Desktop yolu
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

print("Model yükleniyor...")
model = YOLO("yolo11m.pt") # YOLO11 Medium modeli, nesne tespitinde çok daha iyidir.
print("Hazır! SPACE = fotoğraf çek | Q = çıkış")

cache = {}
def translate(word):
    if word in cache:
        return cache[word]
    try:
        result = GoogleTranslator(source="en", target=TARGET_LANGUAGE).translate(word)
        cache[word] = result
        return result
    except:
        return word

def create_photo_card(frame, detections):
    """
    Fotoğraf kartı oluşturur:
    - Beyaz arka plan (Polaroid stili)
    - Üstte fotoğraf
    - Altında tespit edilen kelimeler
    """
    # Fotoğrafı kare yap ve boyutlandır
    h, w = frame.shape[:2]
    size = min(h, w)
    # Ortadan kırp (kare yap)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    cropped = frame[y_start:y_start+size, x_start:x_start+size]
    
    # Fotoğraf boyutu
    PHOTO_SIZE = 500
    photo = cv2.resize(cropped, (PHOTO_SIZE, PHOTO_SIZE))
    
    # BGR → RGB (PIL için)
    photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    photo_pil = Image.fromarray(photo_rgb)
    
    # ── KART BOYUTLARI ──────────────────────────
    PADDING = 30           # Kenar boşluğu
    BOTTOM_SPACE = 160     # Alt yazı alanı
    CARD_W = PHOTO_SIZE + PADDING * 2
    CARD_H = PHOTO_SIZE + PADDING + BOTTOM_SPACE
    
    # Beyaz kart arka planı
    card = Image.new("RGB", (CARD_W, CARD_H), color=(252, 252, 248))
    
    # Hafif gölge efekti (koyu gri dikdörtgen biraz kaydırılmış)
    shadow = Image.new("RGB", (CARD_W, CARD_H), color=(200, 200, 200))
    card.paste(shadow, (4, 4))
    
    # Kartı yeniden oluştur (gölge kartın altına gitsin diye)
    card = Image.new("RGB", (CARD_W + 6, CARD_H + 6), color=(180, 180, 180))
    white_card = Image.new("RGB", (CARD_W, CARD_H), color=(252, 252, 248))
    card.paste(white_card, (0, 0))
    
    # Fotoğrafı karta yapıştır
    card.paste(photo_pil, (PADDING, PADDING))
    
    # Çizim için draw nesnesi
    draw = ImageDraw.Draw(card)
    
    # ── YAZILARI YAZ ───────────────────────────
    # Font yolları (macOS sistem fontları)
    try:
        font_big   = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_tiny  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font_big   = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny  = ImageFont.load_default()
    
    # Alt alan başlangıcı
    text_y = PHOTO_SIZE + PADDING + 15
    
    if detections:
        # En yüksek güven skorlu nesneyi al
        best = max(detections, key=lambda x: x["confidence"])
        
        turkish = best["turkish"]
        english = best["english"]
        confidence = best["confidence"]
        
        # Türkçe kelime (büyük, koyu)
        draw.text(
            (CARD_W // 2, text_y),
            turkish.upper(),
            font=font_big,
            fill=(30, 30, 30),
            anchor="mt"    # middle-top hizalama
        )
        
        # İngilizce kelime (küçük, gri)
        draw.text(
            (CARD_W // 2, text_y + 50),
            english,
            font=font_small,
            fill=(120, 120, 120),
            anchor="mt"
        )
        
        # Güven skoru (en küçük, açık gri)
        draw.text(
            (CARD_W // 2, text_y + 85),
            f"✓ %{int(confidence*100)} doğruluk",
            font=font_tiny,
            fill=(180, 180, 180),
            anchor="mt"
        )
        
        # Diğer tespitler varsa onları da yaz
        if len(detections) > 1:
            others = [d for d in detections if d != best][:2]  # max 2 tane daha
            other_text = "  •  ".join([d["turkish"] for d in others])
            draw.text(
                (CARD_W // 2, text_y + 115),
                other_text,
                font=font_tiny,
                fill=(200, 200, 200),
                anchor="mt"
            )
    else:
        # Hiç nesne bulunamadıysa
        draw.text(
            (CARD_W // 2, text_y + 30),
            "Nesne bulunamadı",
            font=font_small,
            fill=(180, 180, 180),
            anchor="mt"
        )
    
    return card

# ── ANA DÖNGÜ ───────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"HATA: Kamera {CAMERA_INDEX} açılamadı!")
    exit()

photo_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ekranda yardım yazısı göster
    display = frame.copy()
    cv2.putText(display, "SPACE = Fotograf Cek  |  Q = Cikis",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Kamera - Fotograf cekmek icin SPACE", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ── BOŞLUK TUŞU: Fotoğraf çek ──────────────
    if key == ord(' '):
        print("\nFotoğraf çekildi! YOLO analiz ediyor...")
        
        # YOLO ile tespit et
        results = model(frame, conf=CONFIDENCE, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                english_name = model.names[class_id]
                turkish_name = translate(english_name)
                confidence = float(box.conf[0])
                
                detections.append({
                    "english": english_name,
                    "turkish": turkish_name,
                    "confidence": confidence
                })
                
                print(f"  → {english_name} = {turkish_name} (%{int(confidence*100)})")
        
        if not detections:
            print("  Hiç nesne tespit edilemedi.")
        
        # Fotoğraf kartı oluştur
        print("Kart oluşturuluyor...")
        card = create_photo_card(frame, detections)
        
        # Dosya adı: kart_1.png, kart_2.png ...
        photo_count += 1
        if detections:
            best_name = max(detections, key=lambda x: x["confidence"])["turkish"]
            filename = f"{best_name}_{photo_count}.png"
        else:
            filename = f"kart_{photo_count}.png"
        
        save_path = os.path.join(DESKTOP, filename)
        card.save(save_path)
        
        print(f"✅ Kaydedildi: {save_path}")
        
        # Kartı önizleme olarak göster
        card_np = np.array(card)
        card_bgr = cv2.cvtColor(card_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Fotograf Karti - Herhangi bir tusa bas", card_bgr)
        cv2.waitKey(3000)  # 3 saniye göster, sonra kameraya dön
    
    # ── Q: Çıkış ────────────────────────────────
    elif key == ord('q'):
        print("Çıkılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nToplam {photo_count} fotoğraf kartı Desktop'a kaydedildi!")
