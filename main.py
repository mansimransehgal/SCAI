import cv2
import matplotlib.pyplot as plt


# Load the three images
before = cv2.imread("A/train_91.png")
after = cv2.imread("B/train_91.png")
label = cv2.imread("label/train_91.png", cv2.IMREAD_GRAYSCALE)


# Convert OpenCV's BGR format to RGB
before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)


# Display the images
plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(before)
plt.title("Before")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(after)
plt.title("After")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(label, cmap="gray")
plt.title("Actual Change")
plt.axis("off")


plt.tight_layout()
plt.show()