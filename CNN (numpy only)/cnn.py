from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Relu, Softmax  
import json
from utilities.filereader import load_images_from_directory, get_data
from utilities.model import Model
import os
from loss.losses import CategoricalCrossEntropy

import numpy as np
np.random.seed(0)

# find . -name ".DS_Store" -delete --> nếu dính file dư thì xóa
def save_model_structure(model, filename):
    """Lưu cấu trúc mô hình vào file JSON."""
    structure = []
    for layer in model.model:
        layer_info = {
            "type": type(layer).__name__,
            "params": layer.get_params() if hasattr(layer, 'get_params') else {}
        }
        structure.append(layer_info)
    with open(filename, 'w') as f:
        json.dump(structure, f)
    print(f"Model structure saved to {filename}")

def save_weights(model, weights_dir):
    """Lưu trọng số mô hình vào thư mục."""
    os.makedirs(weights_dir, exist_ok=True)
    for i, layer in enumerate(model.model):
        if hasattr(layer, 'save_weights'):
            layer.save_weights(os.path.join(weights_dir, f'layer_{i}.pickle'))
    print(f"Model weights saved to {weights_dir}")

import json
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.activation import Relu, Softmax

def load_model_structure(model_structure_file):
    with open(model_structure_file, 'r') as file:
        model_structure = json.load(file)
    
    layers = []
    for layer in model_structure:
        layer_type = layer['type']
        params = layer['params']
        
        if layer_type == "Convolution":
            # Truyền các tham số cho lớp Convolution
            layers.append(Convolution(**params))
        elif layer_type == "Relu":
            layers.append(Relu())
        elif layer_type == "Pooling":
            # Truyền các tham số cho lớp Pooling
            layers.append(Pooling(**params))
        elif layer_type == "Flatten":
            layers.append(Flatten())
        elif layer_type == "FullyConnected":
            # Truyền các tham số cho lớp FullyConnected
            layers.append(FullyConnected(**params))
        elif layer_type == "Softmax":
            layers.append(Softmax())
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    return layers

def load_weights(model, weights_dir):
    """Tải trọng số cho mô hình từ thư mục."""
    for i, layer in enumerate(model.model):
        weight_file = os.path.join(weights_dir, f'layer_{i}.pickle')
        if hasattr(layer, 'load_weights') and os.path.exists(weight_file):
            layer.load_weights(weight_file)
    print(f"Model weights loaded from {weights_dir}")
    
if __name__ == '__main__':
    # Đọc dữ liệu từ thư mục và chia dữ liệu thành train và test
    train_data, train_labels, test_data, test_labels = get_data(data_path="522H0148_Traffic_detection/Train", num_samples=50000)

    # In thông tin về dữ liệu
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    # Xây dựng mô hình CNN với ReLU cho các lớp ẩn và Softmax cho lớp đầu ra
    model = Model(
        Convolution(filters=16, kernel_shape=(3, 3), padding='same'),
        Relu(),  # Sử dụng ReLU cho lớp ẩn
        Convolution(filters=32, kernel_shape=(3, 3), padding='same'),
        Relu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=128),
        Relu(),
        FullyConnected(units=10),
        Softmax(),  # Sử dụng Softmax cho lớp đầu ra
        name='cnn-traffic-signs'
    )

    # Thiết lập hàm mất mát (loss function)
    model.set_loss(CategoricalCrossEntropy)

    # Huấn luyện mô hình
    model.train(train_data, train_labels, epochs=5, batch_size=256)

    #model.load_weights()
    # Lưu cấu trúc và trọng số mô hình
    save_model_structure(model, "model_structure.json")
    save_weights(model, "model_weights")

    # # Tải lại mô hình để kiểm tra
    # loaded_model = load_model_structure("model_structure.json")
    # load_weights(loaded_model, "model_weights")

    # # Đánh giá độ chính xác trên bộ dữ liệu kiểm tra
    # accuracy = loaded_model.evaluate(test_data, test_labels)
    # print(f'Testing accuracy after loading = {accuracy:.4f}')
