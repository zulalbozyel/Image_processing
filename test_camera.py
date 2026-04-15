import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera {i} bulundu")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Kamera {i}", frame)
            cv2.waitKey(2000)
        cap.release()
    else:
        print(f"Kamera {i} yok")

cv2.destroyAllWindows()
print("Bitti")
