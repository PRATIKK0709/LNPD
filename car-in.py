import cv2
import serial
import time
import numpy as np
import pytesseract
import datetime
import os

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

ser = serial.Serial('/dev/cu.usbmodem1101', 9600)  # Replace with your Arduino's serial port

def activate_webcam_and_capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture a frame.")
        cap.release()
        return

    timestamp = time.strftime("%Y%m%d%H%M%S")
    pics_dir = "./pics"
    os.makedirs(pics_dir, exist_ok=True)
    file_path = os.path.abspath(f"{pics_dir}/{timestamp}.jpg")
    cv2.imwrite(file_path, frame)

    cap.release()
    cv2.destroyAllWindows()

    print(f"Image saved as '{file_path}'")

    process_image(file_path)

    ser.write(b'1')

def process_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Check if the image is loaded successfully
    if img is None:
        print(f"Error: Unable to load image at '{file_path}'")
        return

    img = cv2.resize(img, (600, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Adaptive thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    edged = cv2.Canny(thresh, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)  # Adjust epsilon
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("No contour detected")
        return
    else:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        print(f"Detected license plate Number is: {text}")

        # Get the current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("detected_license_plate.txt", "w") as file:
            file.write(f"{text} | {current_datetime}")

        img = cv2.resize(img, (500, 300))
        Cropped = cv2.resize(Cropped, (400, 200))
        cv2.imshow('car', img)
        cv2.imshow('Cropped', Cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


try:
    while True:
        line = ser.readline()
        if line.strip() == b'ActivateWebcam':
            print("Activating Webcam")
            activate_webcam_and_capture_image()
except KeyboardInterrupt:
    # Graceful exit on keyboard interrupt
    print("Program interrupted by user")
finally:
    ser.close()
