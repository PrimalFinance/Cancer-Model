import pandas as pd
import os 

cwd = os.getcwd()

datasetPath = cwd + '\\Dataset\\cancer.csv'
print(f"{datasetPath}")

#dPath = f"D:\\Coding\\VisualStudioCode\\Projects\\MachineLearning\\CancerModel\\Dateset\\cancer.csv"
#dPath = f"D:\\Coding\\VisualStudioCode\\Projects\\MachineLearning\\CancerModel\\Dataset\\cancer.csv"

# Scikit-learn imports
from sklearn.model_selection import train_test_split
# Pytorch imports
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_size) -> None:
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x



class TorchCancerModel: 
    def __init__(self) -> None:
        self.dataset = pd.DataFrame()
        self.modelParams = {}
    
    
    # ---------------------------------------------------
    def setDataset(self):
        self.dataset = pd.read_csv(datasetPath)
    # ---------------------------------------------------
    def getModelParams(self):
        if self.dataset.empty:
            self.setDataset()
        # Drop diagnosis column. 
        # This is because X is the training data, and the diagnosis is what we are trying to predict. 
        x = self.dataset.drop(columns=['diagnosis(1=m, 0=b)'])
        # Set y to the column we are trying to predict, diagnosis. 
        y = self.dataset['diagnosis(1=m, 0=b)']
        # Split the into training and testing data. 
        # "test_size=0.2" means 80% of the dataset is used for training, 20% is for testing. 
        # The model uses the training portion to learn, then it uses the testing data, which it has never seen before,
        # to evaluate its predictions. 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
        return x_train, x_test, y_train, y_test
        
    # ---------------------------------------------------
    def trainModel(self, epochs = 1000):
        x_train, x_test, y_train, y_test = self.getModelParams()
        model = Network(x_train.shape[1])
        # Convert to array first 
        x_tensor = torch.Tensor(x_train.values)
        y_tensor = torch.Tensor(y_train.values)
        x_train_tensor = torch.FloatTensor(x_tensor)
        y_train_tensor = torch.FloatTensor(y_tensor)
        criterion = nn.BCELoss() # Binary Cross-entropy loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor.view(-1, 1).float())
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
        
    # ---------------------------------------------------
    def evaluateModel(self):
        x_train, x_test, y_train, y_test = self.getModelParams()
        evaluation = self.model.evaluate(x_test, y_test)
        return evaluation
    # ---------------------------------------------------   
    def saveModel(self, filepath='Model/saved-models/tensorflow_cancer_model.h5'):
        if self.model:
            self.model.save(filepath)
            print(f'Model saved to {filepath}')
        else:
            print('No model to save. Train the model first.')
    # ---------------------------------------------------        
    # def loadModel(self, filepath='Model/saved-models/tensorflow_cancer_model.h5'):
    #     if os.path.exists(filepath):
    #         self.model = tf.keras.models.load_model(filepath)
    #         print(f'Model loaded from {filepath}')
    #     else:
    #         print(f'Error: Model file {filepath} not found.')
    
      
    # ---------------------------------------------------
    
    

'''
####
TODO:
Add save and load method for torch model. 
Add evaluation getters 
'''