import cv2
image = cv2.imread("examples/test_image_3.jpg")
resized_image = cv2.resize(image, dsize=(504, 378), interpolation=cv2.INTER_AREA)
cv2.imwrite("examples/test_image_3.jpg", resized_image)