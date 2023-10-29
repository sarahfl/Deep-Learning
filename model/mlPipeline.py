import models

batch = 32
epochs = 50

for batch in range(batch):
    for epoch in range(epochs):
        train, val = models.getTestDataset(
            "Train_Test_Folder/test", batch_size=batch, size_img=(200, 200)
        )
        model = models.loadModelTraining("mobileNetV2Scratch", 200, dropout=0.2)
        history = models.trainModel(model, train, val, epoch, "model")
        plot = models.plotModel(history, "testPlot")
