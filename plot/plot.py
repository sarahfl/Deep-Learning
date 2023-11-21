import csv
import matplotlib.pyplot as plt
import os
import numpy as np

values_to_plot = ["loss", "accuracy", "val_loss", "val_accuracy"]  # loss, accuracy, val_loss, val_accuracy
file_paths = ["mobileNet", "efficientNetB0", "resNet50"]  # Add all file paths here
# file_paths = ["mobileNetNoWeights", "mobileNet26"]  # Add all file paths here

# Initialize lists to store data

for value_to_plot in values_to_plot:
    all_epochs = []
    all_values = []
    labels = []
    for file_path in file_paths:
        # Read data from the CSV file
        epochs = []
        values = []
        with open(f"{file_path}.csv", 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                epochs.append(int(row['epochs']))
                values.append(float(row[value_to_plot]))

        # Append data to lists
        all_epochs.append(epochs)
        all_values.append(values)

        # Extract label from file path
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        labels.append(file_name)

    # Plotting all datasets on the same graph
    plt.figure(figsize=(8, 6))
    for i in range(len(file_paths)):
        plt.plot(all_epochs[i], all_values[i], marker='o', linestyle='-', label=labels[i])

    plt.xlabel('Epochs')
    plt.ylabel(value_to_plot.capitalize())
    plt.title(f'Plot of {value_to_plot.capitalize()} over Epochs')
    plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(min(min(all_epochs)), max(max(all_epochs)) + 1, 1.0))
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(f'{value_to_plot}.png')
    plt.close()
