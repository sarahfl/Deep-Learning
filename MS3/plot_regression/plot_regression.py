import matplotlib.pyplot as plt
import pandas as pd
import os

model_type = "model2_regression"
MODEL_PATH = f"MS3/Model/{model_type}"
FILES = ["history_age", "history_face", "history_gender"]

for file in FILES:
	load_file = f"{os.path.join(MODEL_PATH, file)}.csv"
	data = pd.read_csv(load_file, index_col=0)
	for column in data.columns.tolist():
		plt.plot(data[column], label=column)

	# Adding labels and title
	plt.xlabel('Epochs')
	plt.ylabel('Values')
	plt.title(file)
	plt.legend()

	# Display the plot
	plt.grid(True)
	plt.savefig(f"{os.path.join(MODEL_PATH, file)}_plot.png")
	plt.show()
