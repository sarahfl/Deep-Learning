import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model1 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model1/metrics.csv', index_col=0)
model3 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model3/metrics.csv', index_col=0)
model5 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model5/metrics.csv', index_col=0)
model6 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model6/metrics.csv', index_col=0)

loss1 = model1['Training Loss'].to_numpy()
loss3 = model3['Training Loss'].to_numpy()
loss5 = model5['Training Loss'].to_numpy()
loss6 = model6['Training Loss'].to_numpy()

def plot_arrays(array1, array2, array3, array4):
    epochs = np.arange(len(array1))

    plt.plot(epochs, array1, label='Loss 1')
    plt.plot(epochs, array2, label='Loss 2')
    plt.plot(epochs, array3, label='Loss 3')
    plt.plot(epochs, array4, label='Loss 4')

    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()

    # Plot anzeigen
    plt.show()

plot_arrays(loss1, loss3, loss5, loss6)