import cv2
import numpy as np
import pytesseract
import datetime

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'path to tesseract'

def preprocess_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    return img, gray

def find_license_plate(gray):
    # Detect license plate contours
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt

def recognize_license_plate(gray, screenCnt):
    if screenCnt is None:
        return "No contour detected", None

    # Draw license plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(gray, gray, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    # Extract text from the license plate
    text = pytesseract.image_to_string(cropped, config='--psm 11')
    return text, cropped

def save_result(text):
    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the text to be saved
    output_text = f"{text.strip()} | {current_datetime}"

    # Save the detected license plate number and date/time in a text file
    with open("detected_license_plate.txt", "w") as text_file:
        text_file.write(output_text)

def main():
    image_path = 'audi.jpg'  # Replace with the path to your image
    img, gray = preprocess_image(image_path)
    screenCnt = find_license_plate(gray)
    text, cropped = recognize_license_plate(gray, screenCnt)
    
    # Save the result and display it
    save_result(text)
    print("License Plate Recognition\n")
    print(f"Detected license plate Number is: {text.strip()}")

    # Display the images
    if screenCnt is not None:
        img = cv2.resize(img, (500, 300))
        cropped = cv2.resize(cropped, (400, 200))
        cv2.imshow('Car', img)
        cv2.imshow('Cropped', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
