import os, random, torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

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

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    os.makedirs("artifacts", exist_ok=True)

    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}: test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "artifacts/best.pt")
            print(f"Saved artifacts/best.pt (acc={best_acc:.4f})")

if __name__ == '__main__':
    main()
