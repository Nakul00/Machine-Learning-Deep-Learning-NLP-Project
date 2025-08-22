import torch, argparse
from PIL import Image
from torchvision import transforms
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def load_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return t(img).unsqueeze(0)

def main(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("artifacts/best.pt", map_location=device))
    model.eval()

    x = load_image(image_path).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    print(f"Predicted digit: {pred}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    args = ap.parse_args()
    main(args.image_path)
