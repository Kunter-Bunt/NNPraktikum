import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class WeightVisualizationPlot(object):
    '''
    Here to implement the visualization of the weights
    '''

    def __init__(self, weights):
        self.weights = weights

    def draw_weights(self):
        plt.title("Weights")
        for i in range(self.weights.shape[1]):
            current = self.weights[1:,i]
            img = np.reshape(current, (28,28))
            plt.imshow(img, cmap='Greys_r')
            #plt.imshow(img, cmap='Greys_r', interpolation="nearest")
            if i % 40 == 0:
                plt.show()

        pass
