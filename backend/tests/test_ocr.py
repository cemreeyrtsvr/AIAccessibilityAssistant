import cv2

from camera.camera import Camera
from ocr.ocr_reader import OCRReader

camera = Camera()
ocr = OCRReader()

print("\nOCR Test Started")
print("Press 'R' to read text.")
print("Press 'Q' to quit.\n")

while True:

    frame = camera.read()

    if frame is None:
        break

    cv2.imshow("OCR Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        text = ocr.read_text(frame)

        print("\nDetected Text")
        print("-----------------------")

        if text:
            print(text)
        else:
            print("No text detected.")

        print("-----------------------")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()