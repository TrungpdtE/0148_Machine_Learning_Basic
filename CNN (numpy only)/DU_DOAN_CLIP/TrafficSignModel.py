import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch import nn, optim

class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_id in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_id)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    # Kiểm tra nếu tệp là ảnh (ví dụ .png, .jpg)
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                        image_path = os.path.join(class_dir, filename)
                        self.image_paths.append(image_path)
                        self.labels.append(int(class_id))  # class_id là nhãn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lấy ảnh và nhãn tại vị trí idx
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Định nghĩa các phép biến đổi cho ảnh
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Đường dẫn đến thư mục ảnh
train_dir = "522H0148_Traffic_detection/Train"  # Thư mục chứa các classId và ảnh

# Tạo Dataset và DataLoader
train_dataset = TrafficSignDataset(data_dir=train_dir, transform=transform)

# Kiểm tra lại xem dataset có ảnh hợp lệ không
print(f"Dataset có {len(train_dataset)} mẫu")

# Nếu dataset hợp lệ, tạo DataLoader
if len(train_dataset) > 0:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Kiểm tra dữ liệu
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break  # In 1 batch đầu tiên
else:
    print("Không có ảnh hợp lệ trong dataset.")

# Kiểm tra dữ liệu
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break  # In 1 batch đầu tiên


# Xây dựng mô hình CNN để phân loại biển báo giao thông
class TrafficSignModel(nn.Module):
    def __init__(self):
        super(TrafficSignModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Định nghĩa loss và optimizer
model = TrafficSignModel()
criterion = nn.CrossEntropyLoss()  # Hàm mất mát cho phân loại
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hàm huấn luyện mô hình
def train_model(model, train_loader, criterion, optimizer, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Huấn luyện trên dữ liệu train
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Tiến hành tính toán và backpropagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # In kết quả mỗi epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Hàm đánh giá mô hình trên bộ validation
def validate_model(model, validation_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Huấn luyện mô hình
train_model(model, train_loader, criterion, optimizer, epochs=50)

# Lưu mô hình sau khi huấn luyện
torch.save(model.state_dict(), 'traffic_sign_model.pth')
