import cv2
import numpy as np
import os

# Path to the image
img_path = "data/hard_example/images/123-2-1-29_11_2022-09_21_00_00001344.png"
save_path = "data/hard_example/masks/123-2-1-29_11_2022-09_21_00_00001344.png"

img = cv2.imread(img_path)
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # black canvas

drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), 6, 255, -1)
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw Mask (press 's' to save)")
cv2.setMouseCallback("Draw Mask (press 's' to save)", draw)

display = img.copy()

while True:
    cv2.imshow("Draw Mask (press 's' to save)", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        break
    elif key == ord("r"):
        mask[:] = 0
        display = img.copy()

cv2.destroyAllWindows()

# Save the mask
cv2.imwrite(save_path, mask)
print(f"Mask saved to {save_path}")
