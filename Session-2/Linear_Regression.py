#====Imports======
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#=================

#====DataSet======
X = torch.tensor([[1.0, 1.0],
                  [2.0, 3.0],
                  [3.0, 5.0],
                  [5.0, 2.0],
                  [6.0, 5.0],
                  [7.0, 3.0]], dtype=torch.float32)

Y = torch.tensor([[0.0],   # below line
                  [0.0],
                  [1.0],   # on/above line
                  [1.0],
                  [1.0],
                  [1.0]], dtype=torch.float32)
#=================

#====Model========
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(2,1)  # 2 -> input 1 -> output
        
    def forward(self,x):
        return torch.sigmoid(self.linear(x))  # squashes output between [0,1]
    
model=LinearRegression()
#=================

#====Loss=========
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
#=================

#====Epochs=======
for epoch in range(1000):
    pred=model(X)
    loss=criterion(pred,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%100==0:
        print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")
#==================

#====Testing=======
with torch.no_grad():
    X_test=torch.tensor([[3.0,2.5]])   #Test Item
    print("Prediction: ",model(X_test).item())
    print("Class: ",1 if model(X_test).item()>0.5 else 0)   #Class 1 if model predict >0.5 else Class 0
    
#=================

#====Plotting=====
plt.scatter(X[:,0], X[:,1], c=Y[:,0], cmap="bwr", edgecolors="k")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Classification data")
plt.show()
#=================