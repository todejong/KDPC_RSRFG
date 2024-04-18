import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.io import savemat
from sklearn.model_selection import train_test_split

# Hyper parameters
T_ini = 3
n_basis = 10
in_features = T_ini * 2
out_features = n_basis
N = 15

# System parameters
M = 1  # [kg]
L = 1  # [m]
g = 9.81  # [m/s^2]
b = 0.1  # friction
Ts = 1 / 30  # [s]
J = M * L**2 / 3

# Load multisine data from matlab
mat = scipy.io.loadmat("../KDPC_simulations/Pendulum/data/u_data.mat")
u_data = mat["u_data"]

mat = scipy.io.loadmat("../KDPC_simulations/Pendulum/data/y_data.mat")
y_data = mat["y_data"]

# convert data to tensors
u_data = np.array(u_data)
y_data = np.array(y_data)

# Plot the pendulum trajectory together with the input
fig, axs = plt.subplots(2)
fig.suptitle("Input/output data experiment")
axs[0].plot(y_data)
axs[1].plot(u_data)
axs[0].set(ylabel="theta")
axs[1].set(ylabel="u")
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

# pytorch must be loaded here, otherwise it does not work
import rbf_gauss

u_data = torch.FloatTensor(u_data)
y_data = torch.FloatTensor(y_data)

print(f"u_data = {(u_data).shape}")
print(f"y_data = {y_data.shape}")

U_ini = torch.transpose(u_data[0 : T_ini - 1], 0, 1)
U_0_Nm1 = torch.transpose(u_data[T_ini - 1 : T_ini + N - 1], 0, 1)

Y_ini = torch.transpose(y_data[0:T_ini], 0, 1)
Y_1_N = torch.transpose(y_data[T_ini : T_ini + N], 0, 1)


for i in range(len(y_data) - T_ini - 1 - N):
    if i < 100:
        print(i)
    U_ini = torch.cat((U_ini, torch.transpose(u_data[i + 1 : T_ini + i], 0, 1)), 0)
    u_loop = torch.transpose(u_data[T_ini + i : T_ini + i + N], 0, 1)
    U_0_Nm1 = torch.cat((U_0_Nm1, u_loop), 0)

    Y_ini = torch.cat((Y_ini, torch.transpose(y_data[i + 1 : T_ini + 1 + i], 0, 1)), 0)
    y_loop = torch.transpose(y_data[T_ini + 1 + i : T_ini + 1 + i + N], 0, 1)
    Y_1_N = torch.cat((Y_1_N, y_loop), 0)

X = torch.cat((U_ini, Y_ini, U_0_Nm1), 1)
y = Y_1_N

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=41
# )

# X_train_1 = X_train[:, 0 : (2 * T_ini - 1)]
# X_train_2 = X_train[:, (2 * T_ini - 1) :]

# X_test_1 = X_test[:, 0 : (2 * T_ini - 1)]
# X_test_2 = X_test[:, (2 * T_ini - 1) :]


X1 = X[:, 0 : (2 * T_ini - 1)]
X2 = X[:, (2 * T_ini - 1) :]


# Create the neural network
class Model(nn.Module):
    def __init__(self, in_1_features, out_1_features, in_2_features, out_2_features):
        super().__init__()
        self.l_1 = nn.Linear(in_1_features, out_1_features, bias=False)
        self.l_2 = nn.Linear(out_1_features, out_1_features, bias=False)
        self.l_3 = nn.Linear(in_2_features, out_2_features, bias=False)

    def forward(self, x1, x2):
        x1 = torch.tanh(self.l_1(x1))
        x1 = torch.tanh(self.l_2(x1))
        x = torch.cat((x1, x2), 1)
        x = self.l_3(x)
        return x


# create a manual seed for randomization
torch.manual_seed(41)
# Create an instance of our model
model = Model((2 * T_ini - 1), (n_basis), (n_basis + N), (N))
# Set the criterion for our model to measure the error
criterion = nn.MSELoss()
# Choose Adam optimizer, lr = Learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-5)

print(f"model.parameters() = {model.parameters()}")
# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 2001
losses = []

for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X1, X2)  # get results

    # Measure the loss/error,
    loss = criterion(y_pred, y)

    # keep track of our losses
    losses.append(loss.detach().numpy())

    # print every 10 epoch
    if i % 10 == 0:
        print(f"Epoch: {i} and the loss: {loss}")

    # Do some back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# create a plot
plt.plot(range(epochs), losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()


print(f"model.parameters() = {model.parameters()}")

# evaluate the model on the test data set
# with torch.no_grad():  # turn off backpropagation
#     y_eval = model.forward(X_test_1, X_test_2)  # are features from our test set
#     loss = criterion(y_eval, y_test)  # Find the loss or error

# # print the predicted output and the test data
# print(f"y_eval = {y_eval}")
# print(f"y_test = {y_test}")
# print(f"y_eval-y_test = {y_eval-y_test}")


# print("test 1")
# print(f"model.l_1.parameters {model.l_1.parameters}")

# Convert the model parameters to numpy arrays
weight1 = model.l_1.weight.detach().numpy()
weight2 = model.l_2.weight.detach().numpy()
weight3 = model.l_3.weight.detach().numpy()

print(f"weight1 = {weight1}")


print("test 2")
# save the parameters for use in matlab
# convert parameters to arrays so they can be saved as .mat files
weight1 = {"weight1": weight1}
weight2 = {"weight2": weight2}
weight3 = {"weight3": weight3}
X = {"X": X}
y = {"y": y}

print("test 3")

# save as .mat file
savemat(
    "../KDPC_simulations/Pendulum/data/weight1.mat",
    weight1,
)
savemat(
    "../KDPC_simulations/Pendulum/data/weight2.mat",
    weight2,
)

savemat(
    "../KDPC_simulations/Pendulum/data/weight3.mat",
    weight3,
)


savemat(
    "../KDPC_simulations/Pendulum/data/X.mat",
    X,
)

savemat(
    "../KDPC_simulations/Pendulum/data/y.mat",
    y,
)
# savemat(r"weight.mat", weight)


print(f"u_data = {u_data}")
print(f"u_data.shape = {u_data.shape}")
