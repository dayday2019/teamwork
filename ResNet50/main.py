import argparse
import os
import sys
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import ResNet, Bottleneck


# 数据集
class heartDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.data = pd.read_csv(datapath)
        
    def __len__(self):
        return len(pd.read_csv(self.datapath))

    def __getitem__(self, index):
        id = torch.tensor([int(self.data['id'][index])])
        heart_signals = torch.tensor([float(x) for x in self.data['heartbeat_signals'][index].split(',')])
        if 'label' in self.data.keys():
            label = torch.tensor([int(self.data['label'][index])])
        else:
            label = None

        item = {'id': id,
                'heart_signals': heart_signals,
                'label': label}
        return item
    
    def collate_fn(self, batch):
        id = torch.cat([b['id'] for b in batch])
        heart_signals = torch.stack([b['heart_signals'] for b in batch]).unsqueeze(1)
        label = [b['label'] for b in batch if b['label'] != None]
        if label != []:
            label = torch.cat(label)
        return id, heart_signals, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-save-dir', type=str, default="model")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--cuda', type=str, default='cuda:1')
    args = parser.parse_args()

    ds_train = heartDataset('train.csv')  # 构建训练数据集
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, collate_fn=ds_train.collate_fn)  # 设置batch_size为1，数据随机选取，线程数由控制台参数得到
    ds_test = heartDataset('testB.csv')  # 构建测试数据集
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=ds_test.collate_fn)  # 设置batch_size为1，数据顺序选取，线程数由控制台参数得到

    device = torch.device(args.cuda)
    model = ResNet(Bottleneck,[3,4,6,3])
    model.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    if args.test == 1:
        model.load_state_dict(torch.load('model/model_1.pth'))
        model.to(device)
        progress = tqdm(enumerate(dl_test, 1), total=len(dl_test), ncols=80)
        model.eval()
        submit = []
        acc = 0
        score = 0.0
        for step, (id, heart_signals, label) in progress:
            cls = model(heart_signals.to(device))
            cls = nn.Softmax(dim=1)(cls)
            _, idx = cls.max(1)
            res = [int(id), 0, 0, 0, 0]
            res[int(idx+1)] = 1
            submit.append(res)
            if label != []:
                acc += int((label.to(device) == idx).sum())
                temp = torch.zeros(4)
                temp[int(label)] = 1
                score += float(torch.abs(torch.sub(cls[0], temp.to(device))).sum())
        colu = ["id", "label_0", "label_1", "label_2", "label_3"]
        df = pd.DataFrame(data=submit, columns=colu)
        df.to_csv('submit/test.csv', index=False)
        with open('result.txt', 'a+') as f:
            print('test', file=f)
            if acc != 0:
                print(acc/20000, file=f)
            if score != 0:
                print(score, file=f)
            print('', file=f)
    else:
        model.to(device)
        epochs = 10
        for epoch in range(1, epochs + 1):
            progress = tqdm(enumerate(dl_train, 1), total=len(dl_train), ncols=80)
            model.train()
            for step, (id, heart_signals, label) in progress:
                model.optimizer.zero_grad()
                cls = model(heart_signals.to(device))
                loss = model.loss_func(cls, label.to(device))
                loss.backward()
                model.optimizer.step()
            print(loss)
            if epoch % 1 == 0:
                if not os.path.exists(args.model_save_dir): os.makedirs(args.model_save_dir)
                state_dict = model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, os.path.join(args.model_save_dir, 'model_{}.pth'.format(epoch)))
            progress = tqdm(enumerate(dl_test, 1), total=len(dl_test), ncols=80)
            model.eval()
            submit = []
            acc = 0
            score = 0.0
            for step, (id, heart_signals, label) in progress:
                cls = model(heart_signals.to(device))
                cls = nn.Softmax(dim=1)(cls)
                _, idx = cls.max(1)
                res = [int(id), 0, 0, 0, 0]
                res[int(idx+1)] = 1
                submit.append(res)
                if label != []:
                    acc += int((label.to(device) == idx).sum())
                    temp = torch.zeros(4)
                    temp[int(label)] = 1
                    score += float(torch.abs(torch.sub(cls[0], temp.to(device))).sum())
            colu = ["id", "label_0", "label_1", "label_2", "label_3"]
            df = pd.DataFrame(data=submit, columns=colu)
            df.to_csv('submit/{}.csv'.format(epoch), index=False)
            with open('result.txt', 'a+') as f:
                print(epoch, file=f)
                if acc != 0:
                    print(acc/20000, file=f)
                if score != 0:
                    print(score, file=f)
                print('', file=f)
