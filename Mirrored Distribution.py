import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras import Sequential, layers, losses, optimizers
import os

# Reduce Verbose Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Defining Distribution Strategy
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"],
                                          cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

print(f'Number of Devices: {strategy.num_replicas_in_sync}')

# Extract all images in specified directory into either training or validation datasets
training_dataset = image_dataset_from_directory(
    "train_data",
    validation_split=0.2,
    subset="training",
    image_size=(64, 64),
    color_mode="grayscale",
    seed=96,
    batch_size=32)

validation_dataset = image_dataset_from_directory(
    "train_data",
    validation_split=0.2,
    subset="validation",
    image_size=(64, 64),
    color_mode="grayscale",
    seed=96,
    batch_size=32)


# Defining model
def build_model():
    model = tf.keras.Sequential()
    # Creating convolutional base with data inputs as 64x64 images with 1 color(grayscale)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Dropping a percentage of layer output to prevent over-fitting
    model.add(layers.Dropout(0.45))

    # Creating Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


# Creates a copy of the model for workers in charge of training
with strategy.scope():
    model = build_model()

# Trains the model
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
model.fit(training_dataset,
          validation_data=validation_dataset,
          callbacks=callbacks,
          epochs=20)
