# MNIST Digit Classifier — CNN

Свёрточная нейронная сеть (CNN) на Keras/TensorFlow для распознавания рукописных цифр из датасета MNIST.

## Описание

Проект решает классическую задачу классификации изображений: по картинке 28×28 пикселей с рукописной цифрой (0–9) модель предсказывает, какая это цифра. Используется датасет MNIST (60 000 обучающих и 10 000 тестовых изображений).

## Архитектура модели

- 2× `Conv2D(32)` + `ReLU` + `BatchNormalization` + `MaxPooling2D`
- 2× `Conv2D(64)` + `ReLU` + `BatchNormalization` + `MaxPooling2D`
- `Flatten` + `BatchNormalization`
- `Dense(512, activation="relu")` + `BatchNormalization` + `Dropout(0.3)`
- `Dense(10, activation="softmax")`

Компиляция: `loss="categorical_crossentropy"`, `optimizer="adam"`, `metrics=["accuracy"]`.

## Аугментация данных

Для повышения устойчивости модели используется `ImageDataGenerator`:
- поворот изображений (`rotation_range=7`)
- сдвиг по ширине/высоте (`width_shift_range=0.05`, `height_shift_range=0.07`)
- сдвиг (`shear_range=0.4`)
- масштабирование (`zoom_range=0.05`)

## Обучение

Модель обучается через `model.fit()` на сгенерированных батчах (`batch_size=64`) в течение 10 эпох, с валидацией на тестовой выборке на каждой эпохе.

## Стек технологий

- Python
- TensorFlow / Keras
- Matplotlib (визуализация примеров из датасета)

## Структура

- `MNIST_detection.ipynb` — весь пайплайн: загрузка данных, предобработка, визуализация, построение модели, аугментация, обучение и оценка на тестовой выборке.

## Запуск

```bash
pip install tensorflow matplotlib
jupyter notebook MNIST_detection.ipynb
```

Выполните ячейки по порядку сверху вниз. После обучения точность модели выводится через `model.evaluate(X_test, y_test)`.
