import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(64, activation='tanh'),
    Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.keras')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



# a) StandardScaler służy do skalowania cech, czyli przekształcania wartości cech tak, aby miały średnią równą zero i wariancję równą jeden. Dzięki temu dane są przeskalowane w taki sposób, że poszczególne cechy mają podobne zakresy wartości, co może poprawić wydajność algorytmów uczenia maszynowego. Dane liczbowe są przekształcane poprzez odjęcie średniej wartości cechy i podzielenie przez jej odchylenie standardowe.
#
# b) OneHotEncoder służy do przekształcania etykiet klas na wektory „one-hot”. Oznacza to, że dla każdej etykiety klasowej tworzony jest wektor, w którym wszystkie elementy są równe zero, z wyjątkiem indeksu odpowiadającego danej klasie, który jest równy jeden. Na przykład, dla trzech klas, etykieta klasy 0 zostanie przekształcona na [1, 0, 0], etykieta klasy 1 na [0, 1, 0], a etykieta klasy 2 na [0, 0, 1].
#
# c) Warstwa wejściowa ma tyle neuronów, ile cech jest w danych wejściowych. W tym przypadku X_train.shape[1] oznacza liczbę cech w danych treningowych. Warstwa wyjściowa ma tyle neuronów, ile klas jest w danych wyjściowych. W tym przypadku y_encoded.shape[1] oznacza liczbę klas.
#
# d) Funkcja aktywacji relu (Rectified Linear Unit) jest jedną z popularnych funkcji aktywacji, ale nie zawsze jest najlepsza dla wszystkich problemów. Można eksperymentować z innymi funkcjami aktywacji, takimi jak sigmoid czy tanh, aby sprawdzić, która działa najlepiej dla konkretnego problemu.
#
# e) Wybór optymalizatora, funkcji straty i metryki oceny modelu może mieć wpływ na wyniki uczenia. Różne optymalizatory mogą prowadzić do różnych trajektorii uczenia się, a różne funkcje straty mogą lepiej odzwierciedlać specyfikę problemu. Możemy również dostosować parametry optymalizatora, takie jak szybkość uczenia się, aby zoptymalizować proces uczenia.
#
# f) Rozmiar partii (batch size) określa, ile przykładów jest przetwarzanych jednocześnie podczas jednej iteracji treningowej. Możemy zmienić rozmiar partii, ustawiając parametr batch_size w funkcji model.fit(). Zmiana rozmiaru partii może wpłynąć na stabilność uczenia się oraz czas trwania każdej epoki. Przykładowo, mniejsze rozmiary partii mogą prowadzić do większej zmienności gradientów, co może prowadzić do bardziej niestabilnego uczenia się, ale jednocześnie może być bardziej wydajne obliczeniowo. Natomiast większe rozmiary partii mogą prowadzić do stabilniejszego uczenia się, ale mogą wymagać więcej pamięci.
#
# g) Krzywe uczenia dostarczają informacji na temat wydajności modelu podczas treningu. Na podstawie krzywych uczenia możemy ocenić, jak szybko model się uczy, czy występuje przeuczenie (overfitting) lub niedouczenie (underfitting), oraz kiedy osiągnął najlepszą wydajność. Wydajność sieci neuronowej może być oceniana na podstawie dokładności na zbiorze treningowym i walidacyjnym. Najlepsza wydajność modelu może być osiągnięta wtedy, gdy dokładność na zbiorze walidacyjnym przestaje się poprawiać, a dokładność na zbiorze treningowym pozostaje wysoka.
#
# h) Kod poniżej wykorzystuje model zapisany wcześniej jako plik Keras .keras. Model może być wczytany ponownie i wykorzystany do prognozowania na nowych danych bez konieczności ponownego trenowania. Funkcja load_model z modułu tensorflow.keras.models służy do wczytania zapisanego modelu, a następnie można wykorzystać ten model do prognozowania na nowych danych.


