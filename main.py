
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the images
before = cv2.imread("A/train_91.png")
after = cv2.imread("B/train_91.png")
label = cv2.imread("label/train_91.png", cv2.IMREAD_GRAYSCALE)

# 2. Check that the images loaded correctly
if before is None or after is None or label is None:
    raise FileNotFoundError(
        "Could not load an image. Check the file paths."
    )

# 3. Convert BGR to RGB for correct display
before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

# 4. Ensure both images have matching dimensions
if before.shape != after.shape:
    after = cv2.resize(
        after,
        (before.shape[1], before.shape[0])
    )

# 5. Calculate absolute pixel differences
difference = cv2.absdiff(before, after)

# 6. Convert the colour difference into grayscale
gray_difference = cv2.cvtColor(
    difference, cv2.COLOR_RGB2GRAY
)


# 7. Compare different thresholds
ground_truth = (label > 0).astype(np.uint8)

thresholds = [20, 30, 40, 50, 60, 80, 100]

print("\n--- THRESHOLD EXPERIMENT ---")
print(
    f"{'Threshold':>10} {'Changed %':>12} "
    f"{'Precision':>12} {'Recall':>10} {'F1':>10}"
)

for threshold_value in thresholds:
    _, test_prediction = cv2.threshold(
        gray_difference,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    predicted = (test_prediction > 0).astype(np.uint8)

    tp = np.sum((predicted == 1) & (ground_truth == 1))
    fp = np.sum((predicted == 1) & (ground_truth == 0))
    fn = np.sum((predicted == 0) & (ground_truth == 1))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    changed_percentage = np.mean(predicted) * 100

    print(
        f"{threshold_value:>10} "
        f"{changed_percentage:>11.2f}% "
        f"{precision:>12.4f} "
        f"{recall:>10.4f} "
        f"{f1:>10.4f}"
    )

# 8. Generate our final visualization using threshold 60
threshold_value = 60

_, prediction = cv2.threshold(
    gray_difference,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

# 9. Display the images and prediction
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(before)
plt.title("Before")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(after)
plt.title("After")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(prediction, cmap="gray")
plt.title(f"Prediction (Threshold {threshold_value})")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(label, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()

# 8. Evaluate our prediction against ground truth

# Convert the ground-truth mask into binary values
ground_truth = (label > 0).astype(np.uint8)

# Convert our prediction into binary values
predicted = (prediction > 0).astype(np.uint8)

# Count pixels
true_positive = np.sum(
    (predicted == 1) & (ground_truth == 1)
)

false_positive = np.sum(
    (predicted == 1) & (ground_truth == 0)
)

false_negative = np.sum(
    (predicted == 0) & (ground_truth == 1)
)

# Calculate metrics safely
precision = true_positive / (
    true_positive + false_positive
) if true_positive + false_positive else 0

recall = true_positive / (
    true_positive + false_negative
) if true_positive + false_negative else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

changed_percentage = np.mean(predicted) * 100

print("\n--- CHANGE DETECTION RESULTS ---")
print(f"Predicted changed area: {changed_percentage:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
