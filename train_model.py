from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загружаем MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразуем метки в one-hot формат
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Нормализуем изображения
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Создаем простую нейронную сеть
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Сохраняем обученную модель в формате keras
model.save('trained_model.keras')

# Оцениваем точность на тестовой выборке
accuracy = model.evaluate(x_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')
