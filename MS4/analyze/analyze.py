import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def analyzePromiDataset(path_to_csv):
    df = pd.read_csv(path_to_csv)
    names = df['name'].to_numpy()
    result = dict(Counter(names))
    print(result)

    # Extracting names and occurrences
    names = list(result.keys())
    occurrences = list(result.values())

    # Setting a larger figure size
    plt.figure(figsize=(8, 5))

    # Creating a horizontal bar chart with enough space for names
    plt.barh(names, occurrences, color='blue', height=0.6)
    plt.xlabel('Number of Images')
    plt.ylabel('Names')
    plt.title('Occurrences of People')

    # Adjusting layout to prevent cutting off names
    plt.tight_layout()
    plt.savefig('occurrences.png')
    plt.show()



path_to_csv = '../preprocessing/promis.csv'
analyzePromiDataset(path_to_csv)