import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
from datetime import datetime
import numpy as np

# завантажуємо набір даних з вбудованого датасету mnist і зразу розділяємо  його
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# нормалізуємо дані щоб вони прийняли значення 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# дані у mnist подаються як матриця 28*28 переводимо її у вектор довжиною 784 елементи
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

#модель з 3 шарів
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)), # 128 нейронів розмір вхідного вектора 784 функція активації relu
    layers.Dense(64, activation='relu'), # 64 нейронів функція активації relu
    layers.Dense(10, activation='softmax') # 10 нейронів де кожний нейрон число і функція softmax яка повертає значення 0-1
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# тренування мережі
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


test_loss, test_accuracy = model.evaluate(x_test, y_test)

# далі йде збереження даних у файл щоб візуалізувати графік функції втрат
# так як я використовую Anaconda env для цієї роботи тут відсутній графічний інтерфейс
os.makedirs("training_data", exist_ok=True)

# Дані для збереження
training_history = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss)
}

# Зберігаємо в JSON
json_path = "training_data/mnist_history.json"
with open(json_path, 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"✅ Дані збережено в {json_path}")