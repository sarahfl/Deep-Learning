import models

batch_sizes = [32]
epoch_sizes = [2]
quadratic_image_size = 200

for batch_size in batch_sizes:
    for epoch_size in epoch_sizes:
        train, val = models.get_train_dataset(
            "../Train_Test_Folder/train", batch_size=batch_size, size_img=(quadratic_image_size, quadratic_image_size)
        )

        model = models.load_model_training("mobileNetV2Scratch", quadratic_image_size, dropout=0.2)
        history = models.train_model(model, train, val, epoch_size, "model")
        plot = models.plot_model(history, "testPlot")
