import cv2
import numpy as np
import pytesseract
import datetime
import imutils

pytesseract.pytesseract.tesseract_cmd = r'tesseract path'

# Initialize the video capture from the default camera (usually camera index 0).
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a video file path.

# Initialize a flag to control the loop
exit_key = ord('q')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture video.")
        break

    # Preprocess the frame as you did in your initial code.
    img = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    
    # Apply thresholding to segment the license plate characters from the background.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        # Draw license plate and display results.
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        text = pytesseract.image_to_string(Cropped, config='--psm 8')

        if text.strip():  # Check if the recognized text is not empty
            # Get the current date and time
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create the text to be saved
            output_text = f"{text.strip()}"
            datetimesave = f"{current_datetime}"

            # Save the detected license plate number and date/time in the specified format in a text file
            with open("detected_license_plate.txt", "w") as text_file:
                text_file.write(f"{output_text} | {datetimesave}")

            print("Programming Fever's License Plate Recognition")
            print(f"Detected license plate Number is: {text}")
        
    # Display the original frame.
    cv2.imshow('Car', img)

    # Check for the exit key to quit the loop.
    if cv2.waitKey(1) & 0xFF == exit_key:
        break

cap.release()
cv2.destroyAllWindows()
