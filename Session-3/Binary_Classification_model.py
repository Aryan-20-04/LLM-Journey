#====Imports======
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
#=================

#====DataSet======
X,y=make_moons(n_samples=1000,noise=0.2,random_state=42)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#=================

#====Model========
class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(2,32)  # 2 -> input 1 -> output
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,1)
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.relu(self.l1(x))
        out=self.relu(self.l2(out))
        out=self.l3(out)
        return out
    
model=BCModel()
#=================

#====Loss=========
criterion=nn.BCEWithLogitsLoss()
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
    y_test_pred=model(X_test)
    pred = torch.sigmoid(y_test_pred) > 0.5
    acc=(pred==y_test).sum().item()/y_test.shape[0]
print("Accuracy:",acc)

    
#=================