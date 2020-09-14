import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


from data_reader import get_data
from model import UNet

def main():
    file_paths = "data/files.txt"
    landmark_paths = "data/landmarks.txt"
    cache_path = "cache"
    landmark_wanted = 0
    num_epochs = 100
    log_freq = 1

    x, y = get_data(file_paths, landmark_paths, landmark_wanted, cache_path)
    print(f"Got {len(x)} images with {len(y)} landmarks")
    tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    print("x.shape", tensor_x.shape)
    print("y.shape", tensor_y.shape)

    dataset = TensorDataset(tensor_x,tensor_y)
    dataloader = DataLoader(dataset)

    unet = UNet(in_dim=1, out_dim=6, num_filters=4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)
    
    unet.to(device)

    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            print("inputs.shape", inputs.shape)
            print("labels.shape", labels.shape)
            optimizer.zero_grad()

            outputs = unet(inputs)
            print("outputs", outputs.shape)
            print("labels", labels.shape)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_freq == log_freq-1:  
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / log_freq))
                running_loss = 0.0
    

if __name__ == "__main__":
    main()
