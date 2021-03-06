import torch
import pandas
import numpy as np
from utils import *
from models import *
from tqdm import trange
from sklearn.linear_model import LogisticRegression

best_err = 100.

def rotate(input):
    rotated_imgs = []
    angles = [90, 180, 270]
    for angle in angles:
        rotated_imgs.append(transforms.RandomRotation([angle,angle]))
    rotated_imgs = torch.tensor(rotated_imgs)
    rotation_labels = torch.LongTensor([0,1,2,3])
    return rotated_imgs, rotation_labels

def get_representations(rep, train=True):
    if train:
        dataloader = trainloader
    else:
        dataloader = valloader

    X_ = []
    y_ = []

    for i, (input, target) in enumerate(dataloader):
        output = rep(input)

        X_.append(output.detach().view(output.size(0), -1))
        y_.append(target.detach())

    X_ = [item for sublist in X_ for item in sublist]
    X = torch.stack(X_).numpy()
    y = torch.stack(y_).numpy().flatten()

    return X, y


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
    model = ResNet3()
    trainloader, valloader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],lr=0.1)
    epochs = 5
    t = trange(epochs, desc='error', leave=True)
    for epoch in t:
        t.set_description('error = %d' % best_err)
        t.refresh()
        train_test_pretext()
        train_test_pretext(train=False)

    # fetch and freeze the first two blocks
    rep = nn.Sequential(model.layer1, model.layer2)
    rep.eval()

    # iterate over CIFAR and compile the new dataset
    X_train, y_train  = get_representations(rep)
    X_test, y_test    = get_representations(rep, train=False)
    linear_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X_train,y_train)
    test_acc          = linear_classifier.score(X_test, y_test)
    print("Accuracy: ", test_acc)
