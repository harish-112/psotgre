import cv2
import numpy as np

# Load image
img = cv2.imread('eng_AF_014.jpg')

# Resize for better visualization
img = cv2.resize(img, (1000, int(img.shape[0] * 1000 / img.shape[1])))

# Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# Adaptive Thresholding - better for varying light/background
thresh = cv2.adaptiveThreshold(blurred, 255, 
                               cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY_INV, 
                               15, 10)
cv2.imshow("Adaptive Thresholding", thresh)
cv2.waitKey(0)

# Morphological Closing to connect nearby handwriting
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Wider for handwriting
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Morphological Closing", closed)
cv2.waitKey(0)

# Find contours (potential words)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes
result = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 20 < w < 350 and 10 < h < 120:  # filter small noise or very big boxes
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Segmented Words", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#adhf