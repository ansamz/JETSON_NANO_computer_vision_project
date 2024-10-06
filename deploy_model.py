import torch
import torchvision.transforms as transforms
from PIL import Image

model = torch.jit.load('best_model_traced_TorchScript.pt', map_location=torch.device('cpu'))
model.eval()

print("model eval done")

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    print("predicting")
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print("done predicting")
    return classes[predicted[0].item()]

test_image_path = 'no_image(3).jpg'
prediction = predict(test_image_path)
print(f"Predicted calss is: {prediction}")
