import cv2
import matplotlib.pyplot as plt
import numpy as np

###### Absolute Difference

M1 = cv2.imread('dog1.jpeg', cv2.IMREAD_GRAYSCALE)
M2 = cv2.imread('dog2.jpg', cv2.IMREAD_GRAYSCALE)
M1 = cv2.resize(M1, (250, 250))
M2 = cv2.resize(M2, (250, 250))
plt.imshow(M1, cmap='gray')
plt.title('Image 1')
plt.axis('off')
plt.show()
plt.imshow(M2, cmap='gray')
plt.title('Image 2')
plt.axis('off')
plt.show()


out = cv2.absdiff(M1, M2)
plt.imshow(out, cmap='gray')
plt.title('Absolute Difference')
plt.axis('off')
plt.show()

############ HOG

img = cv2.imread("dog2.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(64,128))
plt.imshow(img,cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

gx = cv2.Sobel(np.float32(img),cv2.CV_32F,1,0,ksize=1)
gy = cv2.Sobel(np.float32(img),cv2.CV_32F,0,1,ksize=1)

magnitude , angle = cv2.cartToPolar(gx,gy,angleInDegrees=True)

cell_size = (8,8)
bin_n = 9
h,w = img.shape
cell_x = w // cell_size[0]
cell_y = h // cell_size[1]

hog_cells = np.zeros((cell_y,cell_x,bin_n),dtype=np.float32)

for i in range(cell_y):
    for j in range(cell_x):
        mag_cell = magnitude[i*8:(i+1)*8, j*8:(j+1)*8]
        ang_cell = angle[i*8:(i+1)*8 , j*8:(j+1)*8]

        hist, _ = np.histogram(ang_cell,bins = bin_n,range = (0,180),weights = mag_cell)
        hog_cells[i,j] = hist

block_size = (2,2)
eps = 1e-5
hog_normalized = []

for i in range(cell_y):
    for j in range(cell_x):
        block = hog_cells[i:i+2 , j:j+2].ravel()
        norm = np.sqrt(np.sum(block)**2 + eps**2)
        block = block // norm
        hog_normalized.append(block)

hog_features = np.hstack(hog_normalized)

print("Hog Feature Vector length : ", len(hog_features))



