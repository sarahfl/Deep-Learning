import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#good
model1 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model1/metrics.csv', index_col=0)
model5 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model5/metrics.csv', index_col=0)
model6 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model6/metrics.csv', index_col=0)

#bad
model4 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model4/metrics.csv', index_col=0)
model8 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model8/metrics.csv', index_col=0)
model12 = pd.read_csv('/home/sarah/Deep-Learning/MS4/triples/Model/model12/metrics.csv', index_col=0)

loss1 = model1['Training Loss'].to_numpy()
loss5 = model5['Training Loss'].to_numpy()
loss6 = model6['Training Loss'].to_numpy()

loss4 = model4['Training Loss'].to_numpy()
loss8 = model8['Training Loss'].to_numpy()
loss12 = model12['Training Loss'].to_numpy()



def plot_loss():
    epochs = np.arange(len(model1))

    plt.plot(epochs, loss1, label='Loss 1')
    plt.plot(epochs, loss5, label='Loss 5')
    plt.plot(epochs, loss6, label='Loss 6')

    plt.plot(epochs, loss4, label='Loss 4')
    plt.plot(epochs, loss8, label='Loss 8')
    plt.plot(epochs, loss12, label='Loss 12')

    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()

    # Plot anzeigen
    plt.savefig('loss_all_model.png')
    plt.show()


plot_loss()
