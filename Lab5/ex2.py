import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History, ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1) # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definicja callback
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history, checkpoint])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()




# a) W preprocessingu dane wejściowe są przetwarzane w celu przygotowania ich do analizy przez sieć neuronową. Funkcja reshape służy do zmiany kształtu danych, w tym przypadku, aby odpowiednio dopasować je do oczekiwanego kształtu danych wejściowych dla modelu. to_categorical jest używane do przekształcenia etykiet kategorialnych (klasy) w postaci binarnej (one-hot encoding), co jest konieczne do szkolenia modelu. np.argmax jest używane do przekształcenia etykiet kategorialnych z powrotem do ich pierwotnej postaci, aby można było porównać przewidywane etykiety z rzeczywistymi.
#
# b) Dane przepływają przez sieć neuronową począwszy od warstwy wejściowej, która przyjmuje dane wejściowe (obrazy cyfr MNIST), następnie przechodzą przez warstwy ukryte, w tym przypadku warstwy konwolucyjne (Conv2D) z funkcją aktywacji ReLU oraz warstwę spłaszczającą (Flatten), a następnie przechodzą przez warstwy w pełni połączone (Dense) z różnymi funkcjami aktywacji, aż do warstwy wyjściowej z funkcją aktywacji softmax, która zwraca prawdopodobieństwa przynależności do każdej z 10 klas (cyfr od 0 do 9). Każda z warstw przekształca dane na podstawie wag, które są optymalizowane podczas procesu uczenia.
#
# c) Najwięcej błędów na macierzy błędów może być między cyframi, które są sobie podobne w pisowni lub kształcie, na przykład między cyfrą 4 a 9, 3 a 8, lub 7 a 2.
#
# d) Krzywe uczenia się pokazują, jak zmieniają się metryki (tutaj dokładność i funkcja straty) w zależności od liczby epok treningu. W tym przypadku krzywe uczenia się wyglądają na stabilne, bez znaków przeuczenia (overfitting) ani niedouczenia się (underfitting). Dokładność zarówno na zbiorze treningowym, jak i walidacyjnym stale rośnie, podobnie jak spadek funkcji straty, co sugeruje, że model nie jest ani za bardzo dopasowany do danych treningowych, ani nie jest zbyt uproszczony.
