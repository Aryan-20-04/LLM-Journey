import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#====Define training data====
X=torch.linspace(-10,10,100).unsqueeze(1)
Y=X**2+2*X+5               #y=x^2+2x+5
#============================

#====Define model=============
model=nn.Sequential(
    nn.Linear(1,32),       #1->input 32->hidden layer
    nn.Tanh(),             #Tanh() is better for polynomials
    nn.Linear(32,1)        #32->hidden layer 1->output
)
#============================

#======Loss Calculation======
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=5e-3)
#============================

#=====Epochs=================
losses=[]
for epoch in range(10000):
    y_pred=model(X)         #Forward pass
    loss=criterion(y_pred,Y)   #Loss calculation
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch%1000==0:
        print(f'Epoch: {epoch}  Loss:{loss.item()}')
        
    losses.append(loss.item())
#===============================
   
#======Plotting=================
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
#==============================

#=====Prediction/Testing=======
with torch.no_grad():
    X_test=torch.tensor([[8.0]])        #Testing for X=8
    y_test=model(X_test)
    print("Predicted Values:",y_test.item())    #Printing the value of y for X=8
    print("Expected Value:",5+2*X_test.item()+X_test.item()**2)

#===============================