import pandas as pd
import matplotlib.pyplot as plt
import os

def plotHistory(path_to_model):
    # Lade die Daten aus den CSV-Dateien
    age_history = pd.read_csv(os.path.join(path_to_model, 'history_age.csv'))
    gender_history = pd.read_csv(os.path.join(path_to_model,'history_gender.csv'))
    face_history = pd.read_csv(os.path.join(path_to_model,'history_face.csv'))

    ## Plot Accuracy
    plt.figure(figsize=(12, 6))

    # Plot für das Age-Output
    plt.plot(age_history['accuracy'], label='Age Accuracy', color='blue')
    plt.plot(age_history['val_accuracy'], label='Age Validation Accuracy', linestyle='dashed', color='blue')

    # Plot für das Gender-Output
    plt.plot(gender_history['accuracy'], label='Gender Accuracy', color='green')
    plt.plot(gender_history['val_accuracy'], label='Gender Validation Accuracy', linestyle='dashed', color='green')

    # Plot für das Face-Output
    plt.plot(face_history['accuracy'], label='Face Accuracy', color='red')
    plt.plot(face_history['val_accuracy'], label='Face Validation Accuracy', linestyle='dashed', color='red')

    plt.title('Accuracy Age, Gender, Face')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')  # Legende oben rechts
    plt.grid(True)
    plt.savefig('accuracy.png')
    plt.show()

    ## Plot Loss
    plt.figure(figsize=(12, 6))

    # Plot für das Age-Output
    plt.plot(age_history['loss'], label='Age Loss', color='blue')
    plt.plot(age_history['val_loss'], label='Age Validation Loss', linestyle='dashed', color='blue')

    # Plot für das Gender-Output
    plt.plot(gender_history['loss'], label='Gender Loss', color='green')
    plt.plot(gender_history['val_loss'], label='Gender Validation Loss', linestyle='dashed', color='green')

    # Plot für das Face-Output
    plt.plot(face_history['loss'], label='Face Loss', color='red')
    plt.plot(face_history['val_loss'], label='Face Validation Loss', linestyle='dashed', color='red')

    plt.title('Loss Age, Gender, Face')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  # Legende oben rechts
    plt.grid(True)

    plt.savefig('loss.png')
    plt.show()

path_to_model = '/home/sarah/Deep-Learning/MS3/Model/model1'
plotHistory(path_to_model)