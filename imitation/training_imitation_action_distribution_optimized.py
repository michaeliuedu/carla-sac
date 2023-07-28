import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter

# tensorboard dev upload --logdir="C:\Users\s1929247\Documents\Ali-Document\Computer Science\Project\imitation\latent_space_imitation\runs\Apr23_02-41-39_pdml3"


device = torch.device("cuda:0")
###Filter on command
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
COMMAND = None
accuracy_threshold=0.03  ##for steering angle
USE_PRETRAINED = False


###importing the data
actions_list = []
images_list = []

with open('imitation_training_data_balanced1.pkl','rb') as af:
    actions_list = pickle.load(af)

with open('imitation_training_images_balanced1.pkl','rb') as f:
    images_list = pickle.load(f)
print('entire training data',len(images_list))

images_list = np.array(images_list)
actions_list = np.array(actions_list)
print('images_list.shape', images_list.shape)
print('actions_list.shape', actions_list.shape)


##creating train dataset
df1 = pd.DataFrame(images_list)
df = df1.iloc[:, :64]


df.loc[:,64] = round(pd.DataFrame(actions_list).iloc[:,0],3)  ##steer
df.loc[:,65] = round(pd.DataFrame(actions_list).iloc[:,1],3)  ##throttle
# df.loc[:,66] = round(pd.DataFrame(actions_list).iloc[:,2],3)  ##brake

for i in range(df.shape[1]):
    df = df.rename(columns={i: 'i'})

X_train  = df.iloc[:, 0:64]  # columns 0-64 for image features
y_train = df.iloc[:, -2:]


df.iloc[:,-3].hist()
df.iloc[:,-2].hist()
df.iloc[:,-1].hist()
print(len(df))

##Model parameters
NUM_FEATURES = len(X_train.columns)


# # Train - Test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,  random_state=69) #stratify=y_train,
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.01,  random_state=69) #stratify=y_val,

##Normalize Input
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

print('X_train', X_train.shape, 'y_train', y_train.shape, 'X_val', X_val.shape, 'y_val', y_val.shape, 'X_test', X_test.shape, 'y_test', y_test.shape)

##Convert Output Variable to Float
y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

##Initialize Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

in_dim = X_train.shape[1]
num_actions = y_train.shape[1]
action_min, action_max = -1, 1
initial_mean_factor = 0.1
initial_std = 0.4
hidden_sizes = (1024, 512, 256, 128)

##Define Neural Network Architecture
class Immitation_nn(nn.Module):
    def __init__(self, in_dim=in_dim, action_min=action_min, hidden_sizes=hidden_sizes, initial_mean_factor=initial_mean_factor, initial_std=initial_std):
        super(Immitation_nn, self).__init__()

        self.imitation_nn = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_actions),nn.Tanh())
        # self.action_mean.weight.data.mul_(initial_mean_factor)
        # self.action_mean.bias.data.fill_(0)
        self.action_logstd = nn.Parameter(torch.full((num_actions,), np.log(initial_std), dtype=torch.float32), requires_grad=True)


    def forward(self, inputs):
        action_mean = self.imitation_nn(inputs)
        action_mean = action_min + ((action_mean + 1) / 2) * (action_max - action_min)
        action_stddev = torch.exp(self.action_logstd)
        return action_mean, action_stddev


    def predict(self, test_inputs):
        x = self.imitation_nn(test_inputs)
        action_mean = action_min + ((x + 1) / 2) * (action_max - action_min)
        return action_mean


def mean_squared_error_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)


# define NLL loss function
def neg_log_likelihood_loss(y_pred, y_true):
    mu, sigma = y_pred
    dist = torch.distributions.Normal(mu, sigma)
    log_prob = dist.log_prob(y_true)
    return -log_prob.mean()

def accuracy(y_pred, y_target, accuracy_threshold=accuracy_threshold):
    diff = torch.abs(y_pred - y_target)
    correct = torch.sum(diff < accuracy_threshold)
    total = y_pred.numel()
    acc = correct / total
    return acc

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

model = Immitation_nn()
model.to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

##Train model
loss_stats = {'train': [], "val": []}
accuracies = {'train': [], "val": []}
rmses = {'train': [], "val": []}

if USE_PRETRAINED:
    model.load_state_dict(torch.load('model_imitation_latent_1action21.pt'))
    min_val_acc = 0.5
else:
    min_val_acc = float("inf")

tb = SummaryWriter()
print("Begin training.")
for epoch in range(1, EPOCHS + 1):
    # TRAINING
    train_loss = 0
    train_acc = 0
    train_rmse = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        loss = neg_log_likelihood_loss(y_train_pred, y_train_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(y_train_pred[0], y_train_batch)
        train_rmse += rmse(y_train_pred[0], y_train_batch)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_rmse /= len(train_loader)

    # VALIDATION
    with torch.no_grad():
        model.eval()
        validation_loss = 0
        val_acc = 0
        val_rmse = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_loss = neg_log_likelihood_loss(y_val_pred, y_val_batch)
            validation_loss += val_loss.item()
            val_acc += accuracy(y_val_pred[0], y_val_batch)
            val_rmse += rmse(y_val_pred[0], y_val_batch)

    validation_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_rmse /= len(val_loader)

    loss_stats['train'].append(train_loss)
    loss_stats['val'].append(validation_loss)
    accuracies['train'].append(train_acc)
    accuracies['val'].append(val_acc)
    rmses['train'].append(train_rmse)
    rmses['val'].append(val_rmse)

    print(f'Epoch {epoch:03}: | Train Loss: {train_loss:.5f} | Val Loss: {validation_loss:.5f} | Train Accuracy: {train_acc:.5f} | Val Accuracy: {val_acc:.5f} | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}')

    tb.add_scalar("train_loss", train_loss, epoch)
    tb.add_scalar("val_loss", validation_loss, epoch)
    tb.add_scalar("train_accuracy", train_acc, epoch)
    tb.add_scalar("val_accuracy", val_acc, epoch)
    tb.add_scalar("train_rmse", train_rmse, epoch)
    tb.add_scalar("val_rmse", val_rmse, epoch)

    if val_acc < min_val_acc:
        min_val_acc = val_acc
        torch.save(model.state_dict(), 'model_imitation_latent_1action21_steer_throttle_optim.pt')
