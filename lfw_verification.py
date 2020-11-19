import nptorch
from nptorch.utils.data import Dataset, DataLoader, ImageFolder
from nptorch.transforms import ToTensor, Resize, Compose, Grayscale
from nptorch import nn
from nptorch.nn import functional as F
from nptorch.optim import SGD
from tqdm import tqdm

trans = Compose([Grayscale(1),
                 Resize((36, 36)),
                 ToTensor()])


class LFWDataset(Dataset):
    def __init__(self, root_dir=None, transform=None):
        super(LFWDataset, self).__init__()
        if root_dir:
            self.data = ImageFolder(root_dir)
        else:
            self.data = None
        self.transform = transform

    def __getitem__(self, index):
        """

        :param index: 表示数据的index
        :return: 返回两张图片与label,label为1表示同一个人，为0表示非同一个人
        """
        img1, label = self.data[2 * index]
        img2 = self.data[2 * index + 1][0]
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, nptorch.tensor(label, dtype=nptorch.float32)

    def __len__(self):
        return len(self.data) // 2


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, dilation=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, dilation=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32 * 6 ** 2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward_once(self, x):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer2(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = nptorch.mean(label * nptorch.pow(euclidean_distance, 2) + (1 - label) * nptorch.pow(
            nptorch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


@nptorch.no_grad()
def test_model(model, test_loader: DataLoader):
    model.eval()
    count = 0
    for img1, img2, lb in test_loader:
        out1, out2 = model(img1, img2)
        dist = F.pairwise_distance(out1, out2)
        p = (dist < 1).int()
        count += (p == lb).float().sum()
    return count.item() / len(test_loader.dataset)


train_root = r'F:\pycharmProjects\FYP-FaceVerification\final_year_project\dataset\lfw_cropped\split_data\train'
siamese_net = SiameseNet()
loss_fcn = ContrastiveLoss()
optimizer = SGD(siamese_net.parameters(), lr=1e-2)
for i in tqdm(range(5)):
    count = 0
    for train_set in tqdm(range(1, 11)):
        train_path = train_root + f'\\0{train_set}'
        dataset = LFWDataset(train_path, trans)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for img1, img2, lb in dataloader:
            count += len(lb)
            print(count)
            siamese_net.train()
            out1, out2 = siamese_net(img1, img2)
            loss = loss_fcn(out1, out2, lb)
            print('loss:', loss)
            loss.backward()
            optimizer.step()
            siamese_net.eval()
            with nptorch.no_grad():
                out1, out2 = siamese_net(img1, img2)
                pred = (F.pairwise_distance(out1, out2) < 1).int()
                print('优化后准确率:', (pred == lb).float().mean().item())
            optimizer.zero_grad()
