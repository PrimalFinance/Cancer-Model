import pandas as pd
import os 

cwd = os.getcwd()

datasetPath = cwd + '\\Dataset\\cancer.csv'
print(f"{datasetPath}")

#dPath = f"D:\\Coding\\VisualStudioCode\\Projects\\MachineLearning\\CancerModel\\Dateset\\cancer.csv"
#dPath = f"D:\\Coding\\VisualStudioCode\\Projects\\MachineLearning\\CancerModel\\Dataset\\cancer.csv"

# Scikit-learn imports
from sklearn.model_selection import train_test_split
# Tensorflow imports
import tensorflow as tf

class CancerModel: 
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
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid')) # Input layer
        model.add(tf.keras.layers.Dense(256, activation='sigmoid')) # Hidden layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # Output layer
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Binary loss function is used since our predictions are binary (benign or malignant).
        model.fit(x_train, y_train, epochs=epochs) # Train model. 
        self.model = model
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
    def loadModel(self, filepath='Model/saved-models/tensorflow_cancer_model.h5'):
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f'Model loaded from {filepath}')
        else:
            print(f'Error: Model file {filepath} not found.')
    
      
    # ---------------------------------------------------