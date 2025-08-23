#====Imports======
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
#=================

#====DataSet======
scaler=StandardScaler()
X,y=make_moons(n_samples=1000,noise=0.2,random_state=42)
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32).view(-1, 1)
y_test=torch.tensor(y_test,dtype=torch.float32).view(-1, 1)
#=================

#====Model========
class LogisticRegression(nn.Module):
    def __init__(self,inps):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(inps,1)
    def forward(self,x):
        return self.linear(x)

model=LogisticRegression(2)
#=================

#====Loss=========
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)
#=================

#====Epochs=======
for epoch in range(1000):
    pred=model(X_train)
    loss=criterion(pred,y_train)
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
    acc = (pred == y_test).sum().item() / y_test.shape[0]
print("Accuracy:", acc)
#=================