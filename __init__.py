from Model.tensorflow_model import TensorCancerModel
from Model.pytorch_model import TorchCancerModel


'''----------------- Tensorflow -----------------'''
def tensorflow(arg):
    if arg == 1:
        trainAndSave()
    elif arg == 2:
        loadAndEvaluate()
def trainAndSave(epochs = 1000):
    model = TensorCancerModel()
    model.trainModel(epochs)
    model.saveModel()
def loadAndEvaluate():
    model = TensorCancerModel()
    model.loadModel()
    evaluation = model.evaluateModel()
    formatted = '{:,.2f}'.format(evaluation[1] * 100)
    print(f"Accuracy: {formatted}%")
    
'''----------------- Pytorch -----------------'''
def pytorch(arg):
    if arg == 1:
        torchTrain()
        
        
def torchTrain():
    model = TorchCancerModel()
    model.trainModel()
    



if __name__ == "__main__":
    #tensorflow(1)
    pytorch(1)
    #loadAndEvaluate()