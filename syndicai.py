from model_building import CNN
import torch
import url_to_image
import torchvision.transforms as transforms

class PythonPredictor:
  
  def __init__(self, config):
    self.model = CNN()
    self.model = self.model.load_state_dict(torch.load("cnn.pth"))
    self._img_size = 16
    self.transform = transforms.Compose([transforms.Resize((self._img_size, self._img_size)), transforms.ToTensor()])

  def predict(self, payload):
    img = url_to_image(payload["url"], "pil")
    img = self.transform(img.convert('L'))
    prediction = torch.max(self.model(img[None, ...]).data,1).indices.item()

    return prediction

