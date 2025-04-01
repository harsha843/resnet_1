import torch
from torch import nn
from torchvision import transforms, models
import streamlit as st
from PIL import Image
import numpy as np

st.header('Food Item Classification')

# Define the class labels (same as data_cat from your original code)
data_cat = ['adhirasam', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch', 'aloo_tikki', 'anarsa', 'ariselu', 
            'bandar_laddu', 'basundi', 'bhatura', 'bhindi_masala', 'biryani', 'boondi', 'butter_chicken', 'chak_hao_kheer', 
            'cham_cham', 'chana_masala', 'chapati', 'chhena_kheeri', 'chicken_razala', 'chicken_tikka', 'chicken_tikka_masala', 
            'chikki', 'daal_baati_churma', 'daal_puri', 'dal_makhani', 'dal_tadka', 'dharwad_pedha', 'doodhpak', 'double_ka_meetha', 
            'dum_aloo', 'gajar_ka_halwa', 'gavvalu', 'ghevar', 'gulab_jamun', 'imarti', 'jalebi', 'kachori', 'kadai_paneer', 
            'kadhi_pakoda', 'kajjikaya', 'kakinada_khaja', 'kalakand', 'karela_bharta', 'kofta', 'kuzhi_paniyaram', 'lassi', 
            'ledikeni', 'litti_chokha', 'lyangcha', 'maach_jhol', 'makki_di_roti_sarson_da_saag', 'malapua', 'misi_roti', 
            'misti_doi', 'modak', 'mysore_pak', 'naan', 'navrattan_korma', 'palak_paneer', 'paneer_butter_masala', 'phirni', 
            'pithe', 'poha', 'poornalu', 'pootharekulu', 'qubani_ka_meetha', 'rabri', 'ras_malai', 'rasgulla', 'sandesh', 
            'shankarpali', 'sheer_korma', 'sheera', 'shrikhand', 'sohan_halwa', 'sohan_papdi', 'sutar_feni', 'unni_appam']

# Load the trained ResNet50 model
model = models.resnet50(pretrained=False)  # Set pretrained=False as you are using custom weights

# Assuming you modified the `fc` layer during training, adjust it accordingly
# For example, if you added layers, you may need to match them here
# If the model used a simple classification layer like:
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(data_cat))
)

# Load the model weights
checkpoint = torch.load('D:\\resnet_1\\resnet50_best_model.pth', map_location=torch.device('cpu'))

model.load_state_dict(checkpoint)  # Load the trained weights

model.eval()  # Set the model to evaluation mode

# Image preprocessing
img_height = 224  # Resizing to the input size expected by ResNet50
img_width = 224
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image:
    img = Image.open(image)

    # Define the transformation pipeline for the input image
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained ResNet normalization values
    ])

    # Apply transformations and add batch dimension
    img_tensor = transform(img).unsqueeze(0)

    # Predict the class
    with torch.no_grad():
        output = model(img_tensor)
        score = torch.nn.functional.softmax(output, dim=1)
        predicted_class = np.argmax(score.numpy())

    # Display the results
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Food item in image is: {data_cat[predicted_class]}")
    st.write(f"With an accuracy of: {np.max(score.numpy()) * 100:.2f}%")
