import os
import argparse
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


from data_reader import get_data
from model import UNet

def get_weigths(data):
    data = np.array(data)
    _, counts = np.unique(data, return_counts=True)
    weights = np.array(counts)/max(counts)
    return torch.Tensor(weights)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_paths',
                        default="data/files.txt")
    parser.add_argument('--landmark_paths',
                        default="data/landmarks.txt")
    parser.add_argument('--landmark', type=int, default=0)
    parser.add_argument('--save_path')
    parser.add_argument('--num_epochs', type=int, default=int(1e9))
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--separator', default=",")
    args = parser.parse_args()

    file_paths = args.file_paths 
    landmark_paths = args.landmark_paths
    landmark_wanted = args.landmark
    num_epochs = args.num_epochs
    log_freq = args.log_freq
    save_path = args.save_path

    x, y = get_data(file_paths, landmark_paths, landmark_wanted, separator=args.separator)
    print(f"Got {len(x)} images with {len(y)} landmarks")
    tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)

    dataset = TensorDataset(tensor_x,tensor_y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    unet = UNet(in_dim=1, out_dim=6, num_filters=4)
    criterion = torch.nn.CrossEntropyLoss(weight=get_weigths(y))
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
        print(f"[{epoch+1}/{num_epochs}] loss: {running_loss}")
        if epoch % log_freq == log_freq-1:  
            if save_path is not None:
                torch.save(unet.state_dict(), os.path.join(save_path, f"unet-{epoch}.pt"))
    

if __name__ == "__main__":
    main()
