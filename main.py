import torch
import pandas
from utils import *
from models import *
from tqdm import tqdm

best_err = 100.

def rotate(input):
    rotated_imgs = []
    angles = [90, 180, 270]
    for angle in angles:
        rotated_imgs.append(transforms.RandomRotation([angle,angle]))
    rotated_imgs = torch.tensor(rotated_imgs)
    rotation_labels = torch.LongTensor([0,1,2,3])
    return rotated_imgs, rotation_labels


def train_test_pretext(train=True):
    global best_err

    if train:
        model.train()
        dataloader = trainloader
    else:
        model.eval()
        dataloader = valloader

    for i, (input, target) in enumerate(dataloader):
        # rotate inputs
        #input, target = rotate(input)
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output, target)

        if train:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            error = get_error(output,target)[0].data[0] # ew
            if error < best_err:
                best_err = error

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dataframe = pd.DataFrame(columns=['epoch','loss','top1','top5'])
    model = LeNet()
    trainloader, valloader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],lr=0.001)
    epochs = 5
    for epoch in tqdm(range(epochs)):
        train_test_pretext()
        train_test_pretext(train=False)

    print(best_err)

    # fetch and freeze the first two blocks

    # plop some linear layers on the end
