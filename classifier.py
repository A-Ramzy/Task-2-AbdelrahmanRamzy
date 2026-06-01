from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATA
iris = load_iris()
X = iris.data
y = iris.target

# 2. SPLIT DATA (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. SCALE THE DATA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. TRAIN THE MODEL (KNN)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. MAKE PREDICTIONS
predictions = model.predict(X_test)

# 6. SEE THE RESULTS
print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, predictions))
print("\n=== FULL REPORT ===")
print(classification_report(y_test, predictions,
      target_names=iris.target_names))