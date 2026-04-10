import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the digits dataset
digits = datasets.load_digits()

# 2. Flatten the images (8x8 images to 64-feature vectors)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 3. Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, random_state=42
)

# 4. Scaling data (Logistic Regression performs better with scaled inputs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create and train the model
# 'multi_class' is set to multinomial for digit classification (0-9)
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)

# 6. Predict on the test set
predicted = clf.predict(X_test)

# 7. Evaluation
print(f"Accuracy Score: {metrics.accuracy_score(y_test, predicted):.4f}")

# 8. Visualization: Plot the first 4 test samples and their predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    # Inverse transform or reshape original data for display
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f"Pred: {prediction}")

plt.show()