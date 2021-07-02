from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
#Hello
# Step1
# Dataset definition
class CSVdataset(Dataset):
    def __init__(self, path):
        # Read the dataset as dataframe using pandas
        df = read_csv(path, header= None)
        # Save the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # Confirm that all the inputs are of the float type
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # Number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    # Return the indices for the train and test rows
    def get_splits(self, n_div=0.33):
        test_size  = round(n_div * len(self.X))
        train_size = len(self.X) - test_size
        # Calculate the split
        return random_split(self, [train_size, test_size])

# Step 2
# Definition of the model
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        #Inputs for the first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        #Inputs for the second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        #Inputs for the third hidden layer
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # Forward propagate the input
    def forward(self, X):
        # Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Input to the second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Input to the third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# Step 3
# Prepare the dataset
def prep_data(path):
    # Load the dataset
    dataset = CSVdataset(path)
    # Divide the dataset into training and test sets
    train, test = dataset.get_splits()
    # Prepare dataloaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test, batch_size= 1024, shuffle=False)
    return train_dl, test_dl

# Step 4
# Train the model
def train_model(train_dl, model):
    # Optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Count the number of epochs
    for epoch in range(100):
        # Count the number of mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # Clear the gradients
            optimizer.zero_grad()
            # Find the model output
            yhat = model(inputs)
            # Find the loss
            loss = criterion(yhat, targets)
            loss.backward()
            # Update model weights
            optimizer.step()

# Step 5
# Test the model
def test_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# Step 6
# Make a prediction for a row of data
def predict(row, model):
    row = Tensor([row])
    # Make the prediction
    yhat = model(row)
    # Retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# Step 7
# Prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prep_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(34)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = test_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
