import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Loading data
diabetes = pd.read_csv("diabetes 1.csv")
X = diabetes.drop(columns=["class"])
y = diabetes["class"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=288493)


# Creating MLP
model = MLPClassifier(hidden_layer_sizes=(6, 8), activation='relu', max_iter=500, random_state=288493)

# Training the model
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model accuracy
mlp_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model (3 layers, 6 neurons): {mlp_accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=["negative", "positive"])
print("Confusion Matrix:")
print(conf_matrix)
# conf_matrix_display.plot()
# plt.show()


# Initialising Classifiers
dd = DecisionTreeClassifier()
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_11 = KNeighborsClassifier(n_neighbors=11)
bayes = GaussianNB()


# Checking classifier results
classifier_results = []
for classifier in [dd, knn_3, knn_5, knn_11, bayes]:
    trained = classifier.fit(X_train, y_train)
    accuracy_score = trained.score(X_test, y_test)
    classifier_results.append((classifier, accuracy_score))
    print(f"{classifier} has accuracy of {accuracy_score * 100:.2f}%")

    predictions = trained.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    confusion_matrix_display = ConfusionMatrixDisplay(
        conf_matrix, display_labels=["negative", "positive"]
    )
    confusion_matrix_display.plot()
    plt.title(f"{classifier} confusion matrix")
    plt.show()


# FN is worse and there are almost 5 times more of them than FP
