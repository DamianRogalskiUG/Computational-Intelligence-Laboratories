import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Loading Data
df = pd.read_csv("iris.csv")

# Splitting dataframe between training and testing sets (70%/30%)
train_set, test_set = train_test_split(df, train_size=0.7, random_state=288493)

unique_classes = df["variety"].unique()


# Splitting sets into train/test inputs and classes
train_inputs = train_set.drop(columns=['variety'])  # Features for training
train_classes = train_set['variety']                # Labels for training
test_inputs = test_set.drop(columns=['variety'])   # Features for testing
test_classes = test_set['variety']                  # Labels for testing


# Initialising Classifiers
dd = DecisionTreeClassifier()
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_11 = KNeighborsClassifier(n_neighbors=11)
bayes = GaussianNB()


# Checking classifier_results
classifier_results = []
for classifier in [dd, knn_3, knn_5, knn_11, bayes]:
    trained = classifier.fit(train_inputs, train_classes)
    accuracy_score = trained.score(test_inputs, test_classes)
    classifier_results.append((classifier, accuracy_score))
    print(f"{classifier} has accuracy of {accuracy_score}")

    predictions = trained.predict(test_inputs)
    conf_matrix = confusion_matrix(test_classes, predictions)
    confusion_matrix_display = ConfusionMatrixDisplay(
        conf_matrix, display_labels=unique_classes
    )
    confusion_matrix_display.plot()
    plt.title(f"{classifier} confusion matrix")
    plt.show()


# Determing the best classifier
best_classifier = max(classifier_results, key=lambda x: x[1])

print(
    f"In my test the {best_classifier[0]} scored the best accuracy of {best_classifier[1]}"
)