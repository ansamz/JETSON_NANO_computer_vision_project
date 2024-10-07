import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.jit.load('best_model_traced_TorchScript.pt', map_location=torch.device('cpu'))
model.eval()

# Define your image transformation
transforming_img = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(image_path):
    # Preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((150, 150))
    image_tensor = transforming_img(image).unsqueeze(0).to(device)

    # Classify the image
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()
        class_name = classes[class_index]
    
    return class_name

def upload_and_classify():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = classify_image(file_path)
        result_label.config(text=f"Predicted class: {result}")
        
        # Display the image
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize image for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference

root = tk.Tk()
root.title("Brain Tumor Detection")

upload_button = tk.Button(root, text="Upload Image", command=upload_and_classify)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Predicted class: None")
result_label.pack(pady=10)

root.mainloop()
