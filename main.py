import numpy as np
from mlxtend.data import loadlocal_mnist
def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

x_train, y_train = loadlocal_mnist(
    images_path=r'D:\PerceptronNN\Dataset\train-images.idx3-ubyte',
    labels_path=r'D:\PerceptronNN\Dataset\train-labels.idx1-ubyte'
)

x_test, y_test = loadlocal_mnist(
    images_path=r'D:\PerceptronNN\Dataset\t10k-images.idx3-ubyte',
    labels_path=r'D:\PerceptronNN\Dataset\t10k-labels.idx1-ubyte'
)

x_train = min_max_normalize(x_train)
x_test = min_max_normalize(x_test)
print("Shape of training data: ", x_train.shape)
print("Shape of testing data: ", y_train.shape)
class DigitPredictor:
    def __init__(self):
        self.outer_layer = np.zeros((10, 784))

        self.bias = np.zeros(10)

    def predict(self, inp):
        output = np.zeros(10)
        for i in range(10):
           output[i] = np.sum(self.outer_layer[i] * inp) + self.bias[i]

        return output

    def train_model(self, train_x, train_y, epochs, alpha):
        for epCount in range(epochs):
            print(f"Current epoch : {epCount+1}")

            for trainCount in range(train_x.shape[0]):
                predicted_value = self.predict(train_x[trainCount])
                predicted_value = np.argmax(predicted_value)



                if(predicted_value!= train_y[trainCount]):
                    self.outer_layer[train_y[trainCount]] += (alpha * train_x[trainCount])
                    self.bias[train_y[trainCount]] += alpha

                else:
                    continue


predictor = DigitPredictor()
predictor.train_model(x_train, y_train, 10, 0.01)

total_correct_predictions = 0
for ind in range(y_test.shape[0]):
    predicted_value = predictor.predict(x_test[ind])

    predicted_value = np.argmax(predicted_value)
    if predicted_value == y_test[ind]:
        total_correct_predictions +=1

precentage = (total_correct_predictions/y_test.shape[0]) *100


print("Total correct prediction percentage: ", precentage)

