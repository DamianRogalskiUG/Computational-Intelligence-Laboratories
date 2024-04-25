import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Loading data
iris = load_iris()
X, y = iris.data, iris.target

# Splitting into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=288493)


# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Liczby to 0, 1 i 2 odpowiadają napisom df.target_names

# Creating mlp
model = MLPClassifier(hidden_layer_sizes=(2), max_iter=4000)

# Training model
model.fit(X_train_scaled, y_train)

# Prediction on the testing set
y_pred = model.predict(X_test_scaled)

# Model's accuracy
model_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model (1 layer, 2 neurons): {model_accuracy:.4f}")


model_3_neurons = MLPClassifier(hidden_layer_sizes=(3), max_iter=4000)
model_3_neurons.fit(X_train_scaled, y_train)
y_pred_3_neurons = model_3_neurons.predict(X_test_scaled)
accuracy_3_neurons = accuracy_score(y_test, y_pred_3_neurons)
print(f"Accuracy of the model (1 layer, 3 neurons): {accuracy_3_neurons:.4f}")


model_2_layers = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=4000)
model_2_layers.fit(X_train_scaled, y_train)
y_pred_2_layers = model_2_layers.predict(X_test_scaled)
accuracy_2_layers = accuracy_score(y_test, y_pred_2_layers)
print(f"Accuracy of the model (2 layers, 3 neurons): {accuracy_2_layers:.4f}")

