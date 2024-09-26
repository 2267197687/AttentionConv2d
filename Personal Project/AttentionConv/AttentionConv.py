import torch
import torch.nn as nn
import torch.nn.functional as F
from d2l import torch as d2l


class LocalSelfAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, square_kernel_size, padding, stride=1):
        super(LocalSelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = square_kernel_size
        self.padding = padding
        self.stride = stride
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        batch_size, _, height, width = x.size()
        C = self.out_channels

        
        query = self.query_conv(x).view(batch_size, C, 1, height, width )
        key = self.key_conv(x)
        value = self.value_conv(x)

        #填充
        key = F.pad(key, (self.padding, self.padding, self.padding, self.padding),
                 mode='constant', value=0)

        value = F.pad(value, (self.padding, self.padding, self.padding, self.padding),
                 mode='constant', value=0)

        #局部注意力窗口
        key_unfold = key.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        key_unfold = key_unfold.contiguous().view(batch_size, C, self.kernel_size**2, height, width )

        value_unfold = value.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        value_unfold = value_unfold.contiguous().view(batch_size, C, self.kernel_size**2, height, width )

        # 用带掩码softmax计算注意力分数
        query = query.permute(0, 3, 4, 2, 1) 
        key_unfold = key_unfold.permute(0 , 3, 4, 1, 2)
        fraction = torch.matmul(query, key_unfold)
        mask = fraction.detach() != 0
        fraction = fraction.masked_fill_(~mask, float('-inf'))
        attention = self.softmax(fraction)

        # 计算加权的值
        value_unfold = value_unfold.permute(0 ,3, 4, 2, 1)
        out = torch.matmul(attention, value_unfold).permute(0, 4, 1, 2, 3).reshape(batch_size, C, height, width)
        
        return F.relu(self.bn(out))


    

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=2), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(LocalSelfAttention2D(64, 64, square_kernel_size=3, padding=1),
                  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) 
b3 = nn.Sequential(LocalSelfAttention2D(128, 128, square_kernel_size=3, padding=1),
                  nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(LocalSelfAttention2D(256, 256, square_kernel_size=3, padding=1),
                   nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(LocalSelfAttention2D(512, 512, square_kernel_size=3, padding=1),
                   nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )
                    


                  
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(1024, 10))

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=128)
def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch{epoch+1}_loss {train_l:.3f}, epoch{epoch+1}_train acc {train_acc:.3f}, '
          f'epoch{epoch+1}_test acc {test_acc:.3f}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
