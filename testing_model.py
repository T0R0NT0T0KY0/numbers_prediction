import cv2
import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image

# Загрузка модели для распознавания цифр
model = load_model('trained_model2.keras')

output_folder = 'dist'

# Создание папки для сохранения изображений цифр
os.makedirs(output_folder, exist_ok=True)


def split_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Преобразование изображения в черно-белый формат
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Нахождение контуров на изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по горизонтальной координате x
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    digit_images = []
    for i, contour in enumerate(contours):
        # Получение координат прямоугольника, ограничивающего контур
        x, y, w, h = cv2.boundingRect(contour)

        # Выделение цифры на изображении
        digit = image[y:y + h, x:x + w]

        # Определите желаемый размер и падинг
        desired_size = 24
        padding = 4

        # Изменение размера изображения до 26x26 (для совместимости с моделью MNIST)
        digit = cv2.resize(digit, (desired_size, desired_size))

        # Добавьте падинг с использованием cv2.copyMakeBorder()
        digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])

        digit = cv2.bitwise_not(digit)

        # Нормализация значений пикселей
        digit = digit / 255.0

        # Сохранение изображения цифры
        digit_filename = os.path.join(output_folder, f'digit_{i}.png')
        cv2.imwrite(digit_filename, digit * 255)  # Умножение на 255 для корректного сохранения

        digit_images.append(digit)

    return digit_images


def predict_number(digit_images):
    predicted_numbers = []
    for i, digit in enumerate(digit_images):
        predicted_numbers.append(predict_digit(f'{output_folder}/digit_{i}.png'))

    return predicted_numbers


def preprocess_image(image_path):
    # Загрузка изображения и преобразование его в формат 28x28
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


def predict_digit(image_path):
    # Подготовка изображения
    processed_image = preprocess_image(image_path)

    # Предсказание с использованием модели
    predictions = model.predict(processed_image)

    # Определение индекса класса с максимальной вероятностью
    predicted_digit = np.argmax(predictions[0])
    print(image_path, predicted_digit)

    return predicted_digit


def main(image_path):
    # Разделение изображения на цифры
    digit_images = split_image(image_path)

    # Предсказание цифр с использованием модели
    predicted_numbers = predict_number(digit_images)

    # Вывод результата
    print("Predicted Numbers:", predicted_numbers)


if __name__ == "__main__":
    image_path = 'img/image.jpg'
    main(image_path)
