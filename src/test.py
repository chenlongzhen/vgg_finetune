import cv2

img = cv2.resize(cv2.imread('../testPic/cat.9000.jpg'), (224, 224))

mean_pixel = [103.939, 116.779, 123.68]
img = img.astype(np.float32, copy=False)
print(img.shape)
print(img)
exit(1)

for c in range(3):
    img[:, :, c] = img[:, :, c] - mean_pixel[c]
img = img.transpose((2,0,1))
img = np.expand_dims(img, axis=0)
