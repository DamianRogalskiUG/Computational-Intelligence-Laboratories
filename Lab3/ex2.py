import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Loading Data
df = pd.read_csv("iris.csv")

# Splitting dataframe between training and testing sets (70%/30%)
train_set, test_set = train_test_split(df, train_size=0.7, random_state=288493)

# Splitting sets into train/test inputs and classes
train_inputs = train_set.drop(columns=['variety'])  # Features for training
train_classes = train_set['variety']                # Labels for training
test_inputs = test_set.drop(columns=['variety'])   # Features for testing
test_classes = test_set['variety']                  # Labels for testing

# Initialisation of a Decision Tree
classifier = DecisionTreeClassifier(random_state=288493)

# Training the tree using fit function
classifier.fit(train_inputs, train_classes)

# Text Visualisation of the Tree
tree_text = export_text(classifier, feature_names=train_inputs.columns.tolist())
print("Decision tree:")
print(tree_text)

# Classifier Evaluation
accuracy = classifier.score(test_inputs, test_classes)
print(f"Classifier Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
predictions = classifier.predict(test_inputs)
cm = confusion_matrix(test_classes, predictions)
print("Confusion matrix:\n", cm)


# AI won by 4%