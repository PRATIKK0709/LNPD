# LNPD
License Plate Recognition using OpenCV and Tesseract OCR

This Python codebase demonstrates license plate recognition using OpenCV for image processing and Tesseract OCR for text extraction. It detects license plates in an image, extracts the license plate number, and saves it along with the date and time of detection in a text file.

## Prerequisites

Before running the code, ensure you have the following prerequisites installed on your system:

- Python (3.x recommended)
- OpenCV (Open Source Computer Vision Library)
- pytesseract (Python wrapper for Tesseract OCR)
- Tesseract OCR

You can install OpenCV, pytesseract, and Tesseract OCR using pip:

```bash
pip install opencv-python-headless
pip install pytesseract
```
For Tesseract OCR, you may need to download and install it separately. Ensure you have the correct path to the Tesseract executable in your code.

## Usage

Clone this repository to your local machine:
```bash
git clone https://github.com/PRATIKK0709/LNPD
cd LNPD
```

Replace ./audi.jpg with the path to the image containing the license plate you want to recognize.
Run the Python script:
```bash
python main.py
```
The code will process the image, detect the license plate, extract the number, and save it in a text file along with the date and time.

