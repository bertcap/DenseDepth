import cv2
image = cv2.imread("examples/bert.jpg")
resized_image = cv2.resize(image, dsize=(1000, 691), interpolation=cv2.INTER_AREA)
cv2.imwrite("examples/bert.jpg", resized_image)