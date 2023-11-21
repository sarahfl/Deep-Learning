from manim import *
import csv
import matplotlib.pyplot as plt
import os

value_to_plot = "accuracy"  # loss, accuracy, val_loss, val_accuracy
file_paths = ["mobileNet", "efficientNetB0", "resNet50"]  # Add all file paths here

class PlotMultipleCSVData(Scene):
    def construct(self):
        # Initialize lists to store data
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

            # Extract label from file path (assuming file name without extension)
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
        plt.tight_layout()

        # Save the plot as an image
        plt.savefig('plot.png')

        # Create an ImageMobject from the saved image
        plot_image = ImageMobject('plot.png')
        plot_image.to_edge(UP)  # Move the image to the top of the screen

        # Display the plot in Manim scene
        self.add(plot_image)
        self.wait(5)  # Display the plot for 5 seconds
