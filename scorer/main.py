import chair_data
from new_model import Classifer, Shared_Classifer, Single_Classifier, Baseline_LeNET
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
from torch import nn

from torch.utils.data import DataLoader

from chair_data import SelfDataset

def train_single(train_loader, test_loader, save_path, ind = 0):
    lr = 1e-3
    num_epochs = 30
    log_interval = 20
    scheduler_interval = 5

    net = Single_Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in trange(1, num_epochs + 1, desc='Epochs', leave=True):
        if epoch % scheduler_interval == scheduler_interval - 1:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train_loss = 0
        net.train()
        for i, data in enumerate(tqdm(train_loader, desc='Batches', leave=False)):
            images, labels = data
            images= images[:, ind, :, :].unsqueeze(dim=1).to(device).float()

            optimizer.zero_grad()
            o = net(images)
            loss = criterion(o, labels.to(device))

            train_loss += loss.item()

            loss.backward()
            optimizer.step()


            if i % log_interval == log_interval - 1:
                print('    Train [%d/%d]\t | \tLoss: %.5f' % (i * o.shape[0], len(train_loader.dataset), loss.item()))
        train_loss = train_loss / i
        print('==> Train | Average loss: %.4f' % train_loss)
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, save_path)

        net.eval()

        total = 0
        correct = 0

        for i, data in enumerate(test_loader):
            images, labels = data
            images = images[:, ind, :, :].unsqueeze(dim=1).to(device).float()

            labels = labels.to(device)
            o = net(images)
            _, predicted = torch.max(o.data,dim=1)
            total += labels.shape[0]

            correct += (labels == predicted).sum().item()


        acc = 100 * (correct / total)
        print('==> Test  | Accuracy: %.4f' % acc)

def train(train_loader, test_loader):
    lr = 1e-3
    num_epochs = 30
    log_interval = 50
    scheduler_interval = 6

    net = Classifer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)




    for epoch in trange(1, num_epochs + 1, desc='Epochs', leave=True):
        if epoch % scheduler_interval == scheduler_interval - 1:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train_loss = 0
        net.train()
        for i, data in enumerate(tqdm(train_loader, desc='Batches', leave=False)):
            images, labels = data
            images_top = images[:, 2, :, :].unsqueeze(dim=1).to(device).float()
            images_side = images[:, 1, :, :].unsqueeze(dim=1).to(device).float()
            images_front = images[:, 0, :, :].unsqueeze(dim=1).to(device).float()

            optimizer.zero_grad()
            o = net(images_top, images_side, images_front)
            loss = criterion(o, labels.to(device))

            train_loss += loss.item()

            loss.backward()
            optimizer.step()


            if i % log_interval == log_interval - 1:
                print('    Train [%d/%d]\t | \tLoss: %.5f' % (i * o.shape[0], len(train_loader.dataset), loss.item()))
        train_loss = train_loss / i
        print('==> Train | Average loss: %.4f' % train_loss)
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'checkpoint.pt')

        net.eval()

        total = 0
        correct = 0

        for i, data in enumerate(test_loader):
            images, labels = data
            images_top = images[:, 2, :, :].unsqueeze(dim=1).to(device).float()
            images_side = images[:, 1, :, :].unsqueeze(dim=1).to(device).float()
            images_front = images[:, 0, :, :].unsqueeze(dim=1).to(device).float()

            labels = labels.to(device)
            o = net(images_top, images_side, images_front)
            _, predicted = torch.max(o.data,dim=1)
            total += labels.shape[0]

            correct += (labels == predicted).sum().item()


        acc = 100 * (correct / total)
        print('==> Test  | Accuracy: %.4f' % acc)




if __name__ == '__main__':


    #data_train_pre = chair_data.ChairDataset(datadir = '/localhome/ama240/Desktop/LeChairs', split = 'train')
    #data_test_pre = chair_data.ChairDataset(datadir='/localhome/ama240/Desktop/LeChairs', split='test')

    data_train = SelfDataset(datadir = '/localhome/ama240/Desktop/764proj/results', split = 'train')
    data_test = SelfDataset(datadir='/localhome/ama240/Desktop/764proj/results', split='test')





    device = torch.device('cuda:0')

    train_loader = DataLoader(data_train, batch_size = 32, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=32, shuffle=True)
    train(train_loader, test_loader)
    #train_single(train_loader, test_loader, ind = 0, save_path = 'side_base.pt')
    #train_single(train_loader, test_loader, ind=1, save_path = 'top_base.pt')
    #train_single(train_loader, test_loader, ind=2, save_path = 'front_base.pt')





