import skimage.feature as sft
import skimage.io as skm
import matplotlib.pyplot as plt

im = skm.imread("dataset/lenna.png", as_gray=True)

lbp = sft.local_binary_pattern(image=im, P=9, R=3)

plt.imshow(lbp, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.show()