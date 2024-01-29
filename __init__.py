from Model.tensorflow_model import TensorCancerModel
from Model.pytorch_model import TorchCancerModel


'''----------------- Tensorflow -----------------'''

class Tflow:
    def __init__(self) -> None:
        self.model = TensorCancerModel()
        
    def trainAndSave(self, epochs = 1000):
        self.model.trainModel(epochs)
        self.model.saveModel()
        
    def loadAndEvaluate(self):
        self.model.loadModel()
        evaluation = self.model.evaluateModel()
        formatted = '{:,.2f}'.format(evaluation[1] * 100)
        print(f"Accuracy: {formatted}%")

'''----------------- Pytorch -----------------'''
class Ptorch:
    def __init__(self) -> None:
        self.model = TorchCancerModel()
        
    def trainAndSave(self):
        self.model.trainModel()
        self.model.saveModel()
        
    def loadAndEvaluate(self):
        self.model.loadModel()
        evaluation = self.model.evaluateModel()
        print(f"Evaluaton: {evaluation}")
        
    
    



if __name__ == "__main__":
    p = Ptorch()
    t = Tflow()
    
    p.loadAndEvaluate()