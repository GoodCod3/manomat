import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar los datos desde los directorios de entrenamiento y validación
train_data = tf.keras.utils.image_dataset_from_directory(
    f"{BASE_DIR}/dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(64, 64),
    batch_size=32,
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    f"{BASE_DIR}/dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(64, 64),
    batch_size=32,
)

# Normalizar las imágenes a valores entre 0 y 1
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 11 clases (0-9 más posiblemente una clase extra)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, validation_data=validation_data, epochs=10)

# Guardar el modelo para usarlo más adelante
model.save('hand_gesture_number_model.h5')

loss, accuracy = model.evaluate(validation_data)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
