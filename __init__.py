from Model.tensorflow_model import CancerModel


def trainAndSave(epochs = 1000):
    model = CancerModel()
    model.trainModel(epochs)
    model.saveModel()
    
def loadAndEvaluate():
    model = CancerModel()
    model.loadModel()
    evaluation = model.evaluateModel()
    formatted = '{:,.2f}'.format(evaluation[1] * 100)
    print(f"Accuracy: {formatted}%")
    


if __name__ == "__main__":
    trainAndSave()
    #loadAndEvaluate()