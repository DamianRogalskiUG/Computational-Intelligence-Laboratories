import itertools
import numpy as np
import matplotlib.pyplot as plt


from keras.src.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


from keras.src.models import Sequential
from keras.src.layers import Flatten, Dense

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Warstwa wyjściowa - binarna klasyfikacja


from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import RMSprop

from sklearn.model_selection import train_test_split
import os

# Ścieżka do folderu z danymi
data_dir = 'dataset_dogs_vs_cats'

# Lista plików z kotami i psami
cats_files = [os.path.join(data_dir, 'cats', file) for file in os.listdir(os.path.join(data_dir, 'cats'))]
dogs_files = [os.path.join(data_dir, 'dogs', file) for file in os.listdir(os.path.join(data_dir, 'dogs'))]

# Oznaczenie kotów jako 0 i psów jako 1
labels = [0] * len(cats_files) + [1] * len(dogs_files)
file_paths = cats_files + dogs_files

# Podział danych na zbiór treningowy i testowy
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=288493)

# Sprawdzenie podziału
print("Liczba zdjęć treningowych:", len(train_files))
print("Liczba zdjęć testowych:", len(test_files))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=5,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=50)


model.save('ex4_model.keras')

# accuracy plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('plots/accuracy_plot.png')  # Save the accuracy plot
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('plots/loss_plot.png')  # Save the loss plot

plt.show()

# model evaluation
score = model.evaluate(validation_generator, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(validation_generator)
#
# # Convert probabilities to binary predictions
# # Convert probabilities to binary predictions
y_pred_binary = np.round(y_pred).flatten().astype(int)

# Compute confusion matrix
conf_matrix = confusion_matrix(validation_generator.classes, y_pred_binary)

# Plot confusion matrix
confusion_matrix_display = ConfusionMatrixDisplay(
    conf_matrix, display_labels=["cat", "dog"]
)
confusion_matrix_display.plot()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')  # Save the confusion matrix plot
plt.show()
