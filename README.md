# Face Detection, Restoration and Classification

Этот проект реализует веб-приложение для обнаружения лиц на изображениях, восстановления лиц (при необходимости)
из профильного вида в фронтальное с помощью GAN и последующей классификацией выражений лица с использованием ResNet152.

---

## Оглавление

- [Описание](#описание)
- [Установка](#установка)
- [Использование](#использование)
- [Архитектура](#архитектура)
- [Модели](#модели)
- [Зависимости](#зависимости)
- [Ссылки на блокноты Google Colab](#СсылкинаблокнотыGoogleColab)

---

## Описание

Приложение позволяет загружать изображения, автоматически обнаруживать лица и определять их ориентацию (фронтальное или
профильное). Для профильных лиц применяется генеративно-состязательная сеть (GAN) для восстановления лица в фронтальное
состояние. Затем для всех лиц выполняется классификация выражения лица с помощью модели ResNet152, обученной на 8 классах
эмоций.

---

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://gitlab.mai.ru/VOKormazova/gan_project/
   cd gan_project
   
2. Создайте и активируйте виртуальное окружение (рекомендуется):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   
4. Убедитесь, что в проекте есть папка haar_cascade_xml/ с файлами каскадов Хаара:
   - haarcascade_frontalface_default.xml
   - haarcascade_profileface.xml

5. Поместите веса моделей в соответствующие папки:
   - weights_rn/resnet152_affectnet.pth
   - weights_gan/generator.pth
   - weights_gan/discriminator.pth


---

## Использование

Запустите приложение Streamlit:
   ```bash
   streamlit run app.py
   ```


В веб-интерфейсе загрузите изображение в формате JPG, JPEG или PNG. Приложение автоматически обнаружит лица,
восстановит профильные лица с помощью GAN и классифицирует выражения лиц.


---

## Архитектура

   - app.py — основной скрипт приложения, реализующий логику загрузки, обнаружения лиц, восстановления и классификации.
   - model_gan.py — содержит архитектуру генератора U-Net с self-attention и PatchGAN дискриминатора для восстановления
     лиц.
   - model_rn.py — содержит модель ResNet152 с дополнительным классификационным слоем для распознавания эмоций.


---

## Модели

1. Генератор GAN (UNetGenerator):
   - U-Net архитектура с канальным и пространственным self-attention.
   - Восстанавливает профильные лица в фронтальное состояние.
2. Дискриминатор GAN (PatchDiscriminator):
   - PatchGAN дискриминатор для оценки качества сгенерированных изображений.
3. Классификатор ResNet152:
   - Предобученная ResNet152 с дообучением на 8 классах эмоций:
       anger, contempt, disgust, fear, happy, neutral, sad, surprise.


---

## Зависимости

Python 3.7+  
torch  
torchvision  
numpy  
opencv-python  
pillow  
streamlit  

---

## Ссылки на блокноты Google Colab

1. https://colab.research.google.com/drive/1y-G_i4d2596s8L3uaydvzVZ7P0gjo8hv?usp=sharing - ResNet152 (классификация эмоций)
2. https://colab.research.google.com/drive/11_SUQEEiXorKkwPFHdYW1zpHlG_9AYao?usp=sharing - U-Net GAN (генерация лиц)
