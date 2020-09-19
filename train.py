import os
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


from data_reader import get_data
from model import UNet

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_paths',
                        default="data/files.txt")
    parser.add_argument('--landmark_paths',
                        default="data/landmarks.txt")
    parser.add_argument('--landmark', type=int, default=0)
    parser.add_argument('--save_path')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=1000)
    args = parser.parse_args()

    file_paths = args.file_paths 
    landmark_paths = args.landmark_paths
    landmark_wanted = args.landmark
    num_epochs = args.num_epochs
    log_freq = args.log_freq
    save_path = args.save_path

    x, y = get_data(file_paths, landmark_paths, landmark_wanted)
    print(f"Got {len(x)} images with {len(y)} landmarks")
    tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)

    dataset = TensorDataset(tensor_x,tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    unet = UNet(in_dim=1, out_dim=6, num_filters=4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)
    
    unet.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = unet(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[{epoch+1}/{num_epochs}] loss: {running_loss}"
        if epoch % log_freq == log_freq-1:  
            if save_path is not None:
                torch.save(unet.state_dict(), os.join(save_path, f"unet-{i}.pt"))
    

if __name__ == "__main__":
    main()
