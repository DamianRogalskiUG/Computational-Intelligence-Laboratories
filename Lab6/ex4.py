import numpy as np
import sns as sns
from keras import Sequential
from keras.src.applications.vgg16 import VGG16
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


train_dir = '../Lab5/dataset_dogs_vs_cats/train'
test_dir = '../Lab5/dataset_dogs_vs_cats/test/'

# Define image generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Adding validation split
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data with validation split
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model
base_model.trainable = False

# Create the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,  # Increase epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Accuracy plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss_plot.png')

plt.show()

# Model evaluation
score = model.evaluate(test_generator, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predictions
y_pred = model.predict(test_generator)
y_pred_binary = np.round(y_pred).flatten().astype(int)

# Compute confusion matrix
conf_matrix = confusion_matrix(test_generator.classes, y_pred_binary)

# Plot confusion matrix
confusion_matrix_display = ConfusionMatrixDisplay(
    conf_matrix, display_labels=["cat", "dog"]
)
confusion_matrix_display.plot()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Finding misclassified images
misclassified_indices = np.where(test_generator.classes != y_pred_binary)[0]
misclassified_images = [test_generator.filepaths[i] for i in misclassified_indices]

print(f"Total misclassified images: {len(misclassified_images)}")

# Display some misclassified images
for i, img_path in enumerate(misclassified_images[:5]):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Misclassified as: {'dog' if y_pred_binary[misclassified_indices[i]] == 1 else 'cat'}")
    plt.show()
