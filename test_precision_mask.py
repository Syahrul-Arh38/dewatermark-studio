import cv2
import numpy as np

img_path = r"C:\Users\Administrator\.gemini\antigravity-ide\brain\f8c36c6f-14f0-4f58-b3f7-b2ddc5a07fe4\.user_uploaded\media_1792302928507.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Let's inspect the bottom-right stamp area:
# x: 480..680, y: 840..1020
crop_stamp = img[840:1020, 480:680]
cv2.imwrite("samples/test_crop_stamp_orig.jpg", crop_stamp)

# Naive block mask (what made it blurry before):
block_mask = np.zeros(crop_stamp.shape[:2], dtype=np.uint8)
cv2.circle(block_mask, (100, 95), 85, 255, -1)
cv2.imwrite("samples/test_crop_stamp_block_mask.png", block_mask)

# Advanced stroke-level mask:
gray_stamp = cv2.cvtColor(crop_stamp, cv2.COLOR_BGR2GRAY)
# 1. Top-hat morphological filter extracts bright strokes smaller than 15px
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
tophat = cv2.morphologyEx(gray_stamp, cv2.MORPH_TOPHAT, k)
_, stroke1 = cv2.threshold(tophat, 16, 255, cv2.THRESH_BINARY)

# 2. Pure bright highlights (letters often have pure bright pixels)
_, stroke2 = cv2.threshold(gray_stamp, 205, 255, cv2.THRESH_BINARY)

# 3. Outer circle ring stroke (3px wide only, NOT solid!)
c_ring = np.zeros(crop_stamp.shape[:2], dtype=np.uint8)
cv2.circle(c_ring, (100, 95), 85, 255, 3)

# 4. Circular bounding ROI mask
circle_roi = np.zeros(crop_stamp.shape[:2], dtype=np.uint8)
cv2.circle(circle_roi, (100, 95), 87, 255, -1)

# Combined stroke mask inside stamp circle:
smart_mask = cv2.bitwise_or(stroke1, stroke2)
smart_mask = cv2.bitwise_or(smart_mask, c_ring)
smart_mask = cv2.bitwise_and(smart_mask, circle_roi)

# Dilate by only 1-2px (to cover anti-aliased edge, without blooming into a solid blob!)
k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
smart_mask = cv2.dilate(smart_mask, k_dil, iterations=1)

cv2.imwrite("samples/test_crop_stamp_smart_mask.png", smart_mask)

# Inpainting comparison using Telea (or LaMa):
inpaint_block = cv2.inpaint(crop_stamp, block_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
inpaint_smart = cv2.inpaint(crop_stamp, smart_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

cv2.imwrite("samples/test_crop_stamp_inpaint_block.jpg", inpaint_block)
cv2.imwrite("samples/test_crop_stamp_inpaint_smart.jpg", inpaint_smart)

print("Block masked pixels:", np.count_nonzero(block_mask))
print("Smart masked pixels:", np.count_nonzero(smart_mask))
print("Smart mask covers only", round(np.count_nonzero(smart_mask) / np.count_nonzero(block_mask) * 100, 1), "% of the block area!")
