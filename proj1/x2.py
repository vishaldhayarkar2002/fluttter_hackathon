import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# Define the input shape for the model
input_shape = (224, 224, 3)

# Create a data generator for the training set
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, 
                                   width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Load the training data from the directory
train_data = train_datagen.flow_from_directory(directory="data/train/", target_size=input_shape[:2],
                                               class_mode='binary', batch_size=32)

# Create a data generator for the validation set

val_datagen = ImageDataGenerator(rescale=1./255)

# Load the validation data from the directory
val_data = val_datagen.flow_from_directory(directory="data/val/", target_size=input_shape[:2], 
                                           class_mode='binary', batch_size=32)

# Load the pre-trained MobileNetV2 model from TensorFlow Hub
base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5", 
                            input_shape=input_shape, trainable=False)

# Create a new sequential model on top of the pre-trained model
model = Sequential([
    base_model,
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
model.fit(train_data, epochs=10, validation_data=val_data)

# Convert the TensorFlow model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('pneumonia_detection.tflite', 'wb') as f:
    f.write(tflite_model)
