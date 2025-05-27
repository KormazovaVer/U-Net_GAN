import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Импорт моделей и загрузка каскадов, моделей (ваш существующий код)
from model_gan import UNetGenerator, PatchDiscriminator
from model_rn import model_rn

root_dir = "haar_cascade_xml/"

face_cascade_frontal = cv2.CascadeClassifier(root_dir + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(root_dir + 'haarcascade_profileface.xml')

class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Автоматический выбор устройства: GPU если доступен, иначе CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем модели на выбранное устройство
model_rn.load_state_dict(torch.load('weights_rn/resnet152_affectnet.pth', map_location=device))
model_rn.to(device)
model_rn.eval()

generator = UNetGenerator()
discriminator = PatchDiscriminator()

generator.load_state_dict(torch.load('weights_gan/generator.pth', map_location=device))
discriminator.load_state_dict(torch.load('weights_gan/discriminator.pth', map_location=device))

generator.to(device)
discriminator.to(device)

generator.eval()
discriminator.eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])


gan_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])


def detect_face_orientation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces_frontal) > 0:
        return 'frontal', faces_frontal
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces_profile) > 0:
        return 'profile', faces_profile
    return None, None


def restore_face_with_gan(face_img):
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    input_tensor = gan_transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        restored_tensor = generator(input_tensor)
    restored_img = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored_img = (restored_img * 255).clip(0, 255).astype(np.uint8)
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
    return restored_img


def classify_face_with_resnet(face_img):
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    input_tensor = resnet_transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_rn(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probs, 3)
    # Возвращаем названия классов и вероятности
    return [(class_names[c], float(p)) for p, c in zip(top_prob[0], top_class[0])]


# Streamlit UI
st.title("Face Detection, Restoration and Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    orientation, faces = detect_face_orientation(image)
    if orientation is None:
        st.warning("Лицо не обнаружено")
    else:
        st.write(f"Обнаружена ориентация лица: {orientation}")
        for i, (x, y, w, h) in enumerate(faces):
            face_img = image[y:y+h, x:x+w]

            if orientation == 'profile':
                restored_face = restore_face_with_gan(face_img)
                st.image(cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB), caption=f"Восстановленное лицо #{i+1}")
                st.success("Профильное лицо восстановлено GAN.")
                # Классификация восстановленного лица
                results = classify_face_with_resnet(restored_face)
                st.write(f"Результаты классификации для восстановленного лица #{i+1} (топ-3):")
                for class_name, prob in results:
                    st.write(f"Класс: {class_name}, Вероятность: {prob:.4f}")
            elif orientation == 'frontal':
                results = classify_face_with_resnet(face_img)
                st.write(f"Результаты классификации для лица #{i+1} (топ-3):")
                for class_name, prob in results:
                    st.write(f"Класс: {class_name}, Вероятность: {prob:.4f}")
