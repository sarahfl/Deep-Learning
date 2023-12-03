import pandas as pd
import matplotlib.pyplot as plt
import os


def plotHistory(path_to_model):
    # Lade die Daten aus den CSV-Dateien
    age_history = pd.read_csv(os.path.join(path_to_model, 'history_age.csv'))
    gender_history = pd.read_csv(os.path.join(path_to_model, 'history_gender.csv'))
    face_history = pd.read_csv(os.path.join(path_to_model, 'history_face.csv'))

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
    plt.legend(loc='lower right')  # Legende oben rechts
    plt.grid(True)
    plt.savefig('/home/sarah/Deep-Learning/MS3/Model/{}/accuracy.png'.format(modelType))
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
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.savefig('/home/sarah/Deep-Learning/MS3/Model/{}/loss.png'.format(modelType))
    plt.show()


def plotBoxplot():
    path_to_model_1 = '/home/sarah/Deep-Learning/MS3/Model/model1_classification'
    path_to_model_2 = '/home/sarah/Deep-Learning/MS3/Model/model2_classification'
    path_to_model_3 = '/home/sarah/Deep-Learning/MS3/Model/model3_classification'
    path_to_model_4 = '/home/sarah/Deep-Learning/MS3/Model/model4_classification'
    path_to_model_5 = '/home/sarah/Deep-Learning/MS3/Model/model5_classification'
    path_to_model_6 = '/home/sarah/Deep-Learning/MS3/Model/model6_classification'
    path_to_model_7 = '/home/sarah/Deep-Learning/MS3/Model/model7_classification'
    path_to_model_8 = '/home/sarah/Deep-Learning/MS3/Model/model8_classification'
    path_to_model_9 = '/home/sarah/Deep-Learning/MS3/Model/model9_classification'
    path_to_model_10 = '/home/sarah/Deep-Learning/MS3/Model/model10_classification'
    path_to_model_11 = '/home/sarah/Deep-Learning/MS3/Model/model11_classification'

    age_history_1 = pd.read_csv(os.path.join(path_to_model_1, 'history_age.csv'))
    gender_history_1 = pd.read_csv(os.path.join(path_to_model_1, 'history_gender.csv'))
    face_history_1 = pd.read_csv(os.path.join(path_to_model_1, 'history_face.csv'))

    age_history_2 = pd.read_csv(os.path.join(path_to_model_2, 'history_age.csv'))
    gender_history_2 = pd.read_csv(os.path.join(path_to_model_2, 'history_gender.csv'))
    face_history_2 = pd.read_csv(os.path.join(path_to_model_2, 'history_face.csv'))

    age_history_3 = pd.read_csv(os.path.join(path_to_model_3, 'history_age.csv'))
    gender_history_3 = pd.read_csv(os.path.join(path_to_model_3, 'history_gender.csv'))
    face_history_3 = pd.read_csv(os.path.join(path_to_model_3, 'history_face.csv'))

    age_history_4 = pd.read_csv(os.path.join(path_to_model_4, 'history_age.csv'))
    gender_history_4 = pd.read_csv(os.path.join(path_to_model_4, 'history_gender.csv'))
    face_history_4 = pd.read_csv(os.path.join(path_to_model_4, 'history_face.csv'))

    age_history_5 = pd.read_csv(os.path.join(path_to_model_5, 'history_age.csv'))
    gender_history_5 = pd.read_csv(os.path.join(path_to_model_5, 'history_gender.csv'))
    face_history_5 = pd.read_csv(os.path.join(path_to_model_5, 'history_face.csv'))

    age_history_6 = pd.read_csv(os.path.join(path_to_model_6, 'history_age.csv'))
    gender_history_6 = pd.read_csv(os.path.join(path_to_model_6, 'history_gender.csv'))
    face_history_6 = pd.read_csv(os.path.join(path_to_model_6, 'history_face.csv'))

    age_history_7 = pd.read_csv(os.path.join(path_to_model_7, 'history_age.csv'))
    gender_history_7 = pd.read_csv(os.path.join(path_to_model_7, 'history_gender.csv'))
    face_history_7 = pd.read_csv(os.path.join(path_to_model_7, 'history_face.csv'))

    age_history_8 = pd.read_csv(os.path.join(path_to_model_8, 'history_age.csv'))
    gender_history_8 = pd.read_csv(os.path.join(path_to_model_8, 'history_gender.csv'))
    face_history_8 = pd.read_csv(os.path.join(path_to_model_8, 'history_face.csv'))

    age_history_9 = pd.read_csv(os.path.join(path_to_model_9, 'history_age.csv'))
    gender_history_9 = pd.read_csv(os.path.join(path_to_model_9, 'history_gender.csv'))
    face_history_9 = pd.read_csv(os.path.join(path_to_model_9, 'history_face.csv'))

    age_history_10 = pd.read_csv(os.path.join(path_to_model_10, 'history_age.csv'))
    gender_history_10 = pd.read_csv(os.path.join(path_to_model_10, 'history_gender.csv'))
    face_history_10 = pd.read_csv(os.path.join(path_to_model_10, 'history_face.csv'))

    age_history_11 = pd.read_csv(os.path.join(path_to_model_11, 'history_age.csv'))
    gender_history_11 = pd.read_csv(os.path.join(path_to_model_11, 'history_gender.csv'))
    face_history_11 = pd.read_csv(os.path.join(path_to_model_11, 'history_face.csv'))

    age_accuracy_1 = age_history_1['val_accuracy'].to_numpy().max()
    age_accuracy_2 = age_history_2['val_accuracy'].to_numpy().max()
    age_accuracy_3 = age_history_3['val_accuracy'].to_numpy().max()
    age_accuracy_4 = age_history_4['val_accuracy'].to_numpy().max()
    age_accuracy_5 = age_history_5['val_accuracy'].to_numpy().max()
    age_accuracy_6 = age_history_6['val_accuracy'].to_numpy().max()
    age_accuracy_7 = age_history_7['val_accuracy'].to_numpy().max()
    age_accuracy_8 = age_history_8['val_accuracy'].to_numpy().max()
    age_accuracy_9 = age_history_9['val_accuracy'].to_numpy().max()
    age_accuracy_10 = age_history_10['val_accuracy'].to_numpy().max()
    age_accuracy_11 = age_history_11['val_accuracy'].to_numpy().max()

    gender_accuracy_1 = gender_history_1['val_accuracy'].to_numpy().max()
    gender_accuracy_2 = gender_history_2['val_accuracy'].to_numpy().max()
    gender_accuracy_3 = gender_history_3['val_accuracy'].to_numpy().max()
    gender_accuracy_4 = gender_history_4['val_accuracy'].to_numpy().max()
    gender_accuracy_5 = gender_history_5['val_accuracy'].to_numpy().max()
    gender_accuracy_6 = gender_history_6['val_accuracy'].to_numpy().max()
    gender_accuracy_7 = gender_history_7['val_accuracy'].to_numpy().max()
    gender_accuracy_8 = gender_history_8['val_accuracy'].to_numpy().max()
    gender_accuracy_9 = gender_history_9['accuracy'].to_numpy().max()
    gender_accuracy_10 = gender_history_10['val_accuracy'].to_numpy().max()
    gender_accuracy_11 = gender_history_11['val_accuracy'].to_numpy().max()

    face_accuracy_1 = face_history_1['val_accuracy'].to_numpy().max()
    face_accuracy_2 = face_history_2['val_accuracy'].to_numpy().max()
    face_accuracy_3 = face_history_3['val_accuracy'].to_numpy().max()
    face_accuracy_4 = face_history_4['val_accuracy'].to_numpy().max()
    face_accuracy_5 = face_history_5['val_accuracy'].to_numpy().max()
    face_accuracy_6 = face_history_6['val_accuracy'].to_numpy().max()
    face_accuracy_7 = face_history_7['val_accuracy'].to_numpy().max()
    face_accuracy_8 = face_history_8['val_accuracy'].to_numpy().max()
    face_accuracy_9 = face_history_9['val_accuracy'].to_numpy().max()
    face_accuracy_10 = face_history_10['val_accuracy'].to_numpy().max()
    face_accuracy_11 = face_history_11['val_accuracy'].to_numpy().max()

    age_loss_1 = age_history_1['val_loss'].to_numpy().min()
    age_loss_2 = age_history_2['val_loss'].to_numpy().min()
    age_loss_3 = age_history_3['val_loss'].to_numpy().min()
    age_loss_4 = age_history_4['val_loss'].to_numpy().min()
    age_loss_5 = age_history_5['val_loss'].to_numpy().min()
    age_loss_6 = age_history_6['val_loss'].to_numpy().min()
    age_loss_7 = age_history_7['val_loss'].to_numpy().min()
    age_loss_8 = age_history_8['val_loss'].to_numpy().min()
    age_loss_9 = age_history_9['val_loss'].to_numpy().min()
    age_loss_10 = age_history_10['val_loss'].to_numpy().min()
    age_loss_11 = age_history_11['val_loss'].to_numpy().min()

    gender_loss_1 = gender_history_1['val_loss'].to_numpy().min()
    gender_loss_2 = gender_history_2['val_loss'].to_numpy().min()
    gender_loss_3 = gender_history_3['val_loss'].to_numpy().min()
    gender_loss_4 = gender_history_4['val_loss'].to_numpy().min()
    gender_loss_5 = gender_history_5['val_loss'].to_numpy().min()
    gender_loss_6 = gender_history_6['val_loss'].to_numpy().min()
    gender_loss_7 = gender_history_7['val_loss'].to_numpy().min()
    gender_loss_8 = gender_history_8['val_loss'].to_numpy().min()
    gender_loss_9 = gender_history_9['val_loss'].to_numpy().min()
    gender_loss_10 = gender_history_10['val_loss'].to_numpy().min()
    gender_loss_11 = gender_history_11['val_loss'].to_numpy().min()

    face_loss_1 = face_history_1['val_loss'].to_numpy().min()
    face_loss_2 = face_history_2['val_loss'].to_numpy().min()
    face_loss_3 = face_history_3['val_loss'].to_numpy().min()
    face_loss_4 = face_history_4['val_loss'].to_numpy().min()
    face_loss_5 = face_history_5['val_loss'].to_numpy().min()
    face_loss_6 = face_history_6['val_loss'].to_numpy().min()
    face_loss_7 = face_history_7['val_loss'].to_numpy().min()
    face_loss_8 = face_history_8['val_loss'].to_numpy().min()
    face_loss_9 = face_history_9['val_loss'].to_numpy().min()
    face_loss_10 = face_history_10['val_loss'].to_numpy().min()
    face_loss_11 = face_history_11['val_loss'].to_numpy().min()

    data_accuracy = [[age_accuracy_1, age_accuracy_2, age_accuracy_3, age_accuracy_4, age_accuracy_5, age_accuracy_6,
                      age_accuracy_7, age_accuracy_8, age_accuracy_9, age_accuracy_10],
                     [gender_accuracy_1, gender_accuracy_2, gender_accuracy_3, gender_accuracy_4, gender_accuracy_5,
                      gender_accuracy_6, gender_accuracy_7, gender_accuracy_8, gender_accuracy_9, gender_accuracy_10],
                     [face_accuracy_1, face_accuracy_2, face_accuracy_3, face_accuracy_4, face_accuracy_5,
                      face_accuracy_6, face_accuracy_7, face_accuracy_8, face_accuracy_9, face_accuracy_10]]

    # Boxplot
    outliers = dict(markerfacecolor='r', marker='o', markersize=8, linestyle='none')
    plt.boxplot(data_accuracy, labels=['Age', 'Gender', 'Face'], flierprops=outliers)
    # plt.xlabel('Parameter Sets')

    plt.plot(1, age_accuracy_11, 'o', color='green')
    plt.plot(2, gender_accuracy_11, 'bo')
    plt.plot(3, face_accuracy_11, 'ro')
    plt.ylabel('Accuracy')
    plt.title('Accuracy für Age, Gender und Face')
    plt.savefig('boxplot_accuracy.png')
    plt.show()

    data_loss = [[age_loss_1, age_loss_2, age_loss_3, age_loss_4, age_loss_5, age_loss_6, age_loss_7, age_loss_8, age_loss_9, age_loss_10],
                 [gender_loss_1, gender_loss_2, gender_loss_3, gender_loss_4, gender_loss_5, gender_loss_6,
                  gender_loss_7, gender_loss_8, gender_loss_9, gender_loss_10],
                 [face_loss_1, face_loss_2, face_loss_3, face_loss_4, face_loss_5, face_loss_6, face_loss_7,
                  face_loss_8, face_loss_9, face_loss_10]
                 ]

    print(data_loss)
    outliers = dict(markerfacecolor='r', marker='o', markersize=8, linestyle='none')
    plt.boxplot(data_loss, labels=['Age', 'Gender', 'Face'], flierprops=outliers)
    # plt.xlabel('Parameter Sets')



    # Zusätzlichen Punkt hinzufügen

    plt.plot(1, age_loss_11, 'o', color='green')
    plt.plot(2, gender_loss_11, 'bo')
    plt.plot(3, face_loss_11, 'ro')

    plt.ylabel('Loss')
    plt.title('Loss für Age, Gender und Face')
    plt.savefig('boxplot_loss.png')
    plt.show()


def accuracy_loss_combination():
    path_to_model_1 = '/home/sarah/Deep-Learning/MS3/Model/model1_classification'
    path_to_model_2 = '/home/sarah/Deep-Learning/MS3/Model/model2_classification'
    path_to_model_3 = '/home/sarah/Deep-Learning/MS3/Model/model3_classification'
    path_to_model_4 = '/home/sarah/Deep-Learning/MS3/Model/model4_classification'
    path_to_model_5 = '/home/sarah/Deep-Learning/MS3/Model/model5_classification'
    path_to_model_6 = '/home/sarah/Deep-Learning/MS3/Model/model6_classification'
    path_to_model_7 = '/home/sarah/Deep-Learning/MS3/Model/model7_classification'
    path_to_model_8 = '/home/sarah/Deep-Learning/MS3/Model/model8_classification'
    path_to_model_9 = '/home/sarah/Deep-Learning/MS3/Model/model9_classification'
    path_to_model_10 = '/home/sarah/Deep-Learning/MS3/Model/model10_classification'
    path_to_model_11 = '/home/sarah/Deep-Learning/MS3/Model/model11_classification'

    age_history_1 = pd.read_csv(os.path.join(path_to_model_1, 'history_age.csv'))
    gender_history_1 = pd.read_csv(os.path.join(path_to_model_1, 'history_gender.csv'))
    face_history_1 = pd.read_csv(os.path.join(path_to_model_1, 'history_face.csv'))

    age_history_2 = pd.read_csv(os.path.join(path_to_model_2, 'history_age.csv'))
    gender_history_2 = pd.read_csv(os.path.join(path_to_model_2, 'history_gender.csv'))
    face_history_2 = pd.read_csv(os.path.join(path_to_model_2, 'history_face.csv'))

    age_history_3 = pd.read_csv(os.path.join(path_to_model_3, 'history_age.csv'))
    gender_history_3 = pd.read_csv(os.path.join(path_to_model_3, 'history_gender.csv'))
    face_history_3 = pd.read_csv(os.path.join(path_to_model_3, 'history_face.csv'))

    age_history_4 = pd.read_csv(os.path.join(path_to_model_4, 'history_age.csv'))
    gender_history_4 = pd.read_csv(os.path.join(path_to_model_4, 'history_gender.csv'))
    face_history_4 = pd.read_csv(os.path.join(path_to_model_4, 'history_face.csv'))

    age_history_5 = pd.read_csv(os.path.join(path_to_model_5, 'history_age.csv'))
    gender_history_5 = pd.read_csv(os.path.join(path_to_model_5, 'history_gender.csv'))
    face_history_5 = pd.read_csv(os.path.join(path_to_model_5, 'history_face.csv'))

    age_history_6 = pd.read_csv(os.path.join(path_to_model_6, 'history_age.csv'))
    gender_history_6 = pd.read_csv(os.path.join(path_to_model_6, 'history_gender.csv'))
    face_history_6 = pd.read_csv(os.path.join(path_to_model_6, 'history_face.csv'))

    age_history_7 = pd.read_csv(os.path.join(path_to_model_7, 'history_age.csv'))
    gender_history_7 = pd.read_csv(os.path.join(path_to_model_7, 'history_gender.csv'))
    face_history_7 = pd.read_csv(os.path.join(path_to_model_7, 'history_face.csv'))

    age_history_8 = pd.read_csv(os.path.join(path_to_model_8, 'history_age.csv'))
    gender_history_8 = pd.read_csv(os.path.join(path_to_model_8, 'history_gender.csv'))
    face_history_8 = pd.read_csv(os.path.join(path_to_model_8, 'history_face.csv'))

    age_history_9 = pd.read_csv(os.path.join(path_to_model_9, 'history_age.csv'))
    gender_history_9 = pd.read_csv(os.path.join(path_to_model_9, 'history_gender.csv'))
    face_history_9 = pd.read_csv(os.path.join(path_to_model_9, 'history_face.csv'))

    age_history_10 = pd.read_csv(os.path.join(path_to_model_10, 'history_age.csv'))
    gender_history_10 = pd.read_csv(os.path.join(path_to_model_10, 'history_gender.csv'))
    face_history_10 = pd.read_csv(os.path.join(path_to_model_10, 'history_face.csv'))

    age_history_11 = pd.read_csv(os.path.join(path_to_model_11, 'history_age.csv'))
    gender_history_11 = pd.read_csv(os.path.join(path_to_model_11, 'history_gender.csv'))
    face_history_11 = pd.read_csv(os.path.join(path_to_model_11, 'history_face.csv'))

    age_accuracy_1 = age_history_1['val_accuracy'].to_numpy().max()
    age_accuracy_2 = age_history_2['val_accuracy'].to_numpy().max()
    age_accuracy_3 = age_history_3['val_accuracy'].to_numpy().max()
    age_accuracy_4 = age_history_4['val_accuracy'].to_numpy().max()
    age_accuracy_5 = age_history_5['val_accuracy'].to_numpy().max()
    age_accuracy_6 = age_history_6['val_accuracy'].to_numpy().max()
    age_accuracy_7 = age_history_7['val_accuracy'].to_numpy().max()
    age_accuracy_8 = age_history_8['val_accuracy'].to_numpy().max()
    age_accuracy_9 = age_history_9['val_accuracy'].to_numpy().max()
    age_accuracy_10 = age_history_10['val_accuracy'].to_numpy().max()
    age_accuracy_11 = age_history_11['val_accuracy'].to_numpy().max()

    gender_accuracy_1 = gender_history_1['val_accuracy'].to_numpy().max()
    gender_accuracy_2 = gender_history_2['val_accuracy'].to_numpy().max()
    gender_accuracy_3 = gender_history_3['val_accuracy'].to_numpy().max()
    gender_accuracy_4 = gender_history_4['val_accuracy'].to_numpy().max()
    gender_accuracy_5 = gender_history_5['val_accuracy'].to_numpy().max()
    gender_accuracy_6 = gender_history_6['val_accuracy'].to_numpy().max()
    gender_accuracy_7 = gender_history_7['val_accuracy'].to_numpy().max()
    gender_accuracy_8 = gender_history_8['val_accuracy'].to_numpy().max()
    gender_accuracy_9 = gender_history_9['accuracy'].to_numpy().max()
    gender_accuracy_10 = gender_history_10['val_accuracy'].to_numpy().max()
    gender_accuracy_11 = gender_history_11['val_accuracy'].to_numpy().max()

    face_accuracy_1 = face_history_1['val_accuracy'].to_numpy().max()
    face_accuracy_2 = face_history_2['val_accuracy'].to_numpy().max()
    face_accuracy_3 = face_history_3['val_accuracy'].to_numpy().max()
    face_accuracy_4 = face_history_4['val_accuracy'].to_numpy().max()
    face_accuracy_5 = face_history_5['val_accuracy'].to_numpy().max()
    face_accuracy_6 = face_history_6['val_accuracy'].to_numpy().max()
    face_accuracy_7 = face_history_7['val_accuracy'].to_numpy().max()
    face_accuracy_8 = face_history_8['val_accuracy'].to_numpy().max()
    face_accuracy_9 = face_history_9['val_accuracy'].to_numpy().max()
    face_accuracy_10 = face_history_10['val_accuracy'].to_numpy().max()
    face_accuracy_11 = face_history_11['val_accuracy'].to_numpy().max()

    age_loss_1 = age_history_1['val_loss'].to_numpy().min()
    age_loss_2 = age_history_2['val_loss'].to_numpy().min()
    age_loss_3 = age_history_3['val_loss'].to_numpy().min()
    age_loss_4 = age_history_4['val_loss'].to_numpy().min()
    age_loss_5 = age_history_5['val_loss'].to_numpy().min()
    age_loss_6 = age_history_6['val_loss'].to_numpy().min()
    age_loss_7 = age_history_7['val_loss'].to_numpy().min()
    age_loss_8 = age_history_8['val_loss'].to_numpy().min()
    age_loss_9 = age_history_9['val_loss'].to_numpy().min()
    age_loss_10 = age_history_10['val_loss'].to_numpy().min()
    age_loss_11 = age_history_11['val_loss'].to_numpy().min()

    gender_loss_1 = gender_history_1['val_loss'].to_numpy().min()
    gender_loss_2 = gender_history_2['val_loss'].to_numpy().min()
    gender_loss_3 = gender_history_3['val_loss'].to_numpy().min()
    gender_loss_4 = gender_history_4['val_loss'].to_numpy().min()
    gender_loss_5 = gender_history_5['val_loss'].to_numpy().min()
    gender_loss_6 = gender_history_6['val_loss'].to_numpy().min()
    gender_loss_7 = gender_history_7['val_loss'].to_numpy().min()
    gender_loss_8 = gender_history_8['val_loss'].to_numpy().min()
    gender_loss_9 = gender_history_9['val_loss'].to_numpy().min()
    gender_loss_10 = gender_history_10['val_loss'].to_numpy().min()
    gender_loss_11 = gender_history_11['val_loss'].to_numpy().min()

    face_loss_1 = face_history_1['val_loss'].to_numpy().min()
    face_loss_2 = face_history_2['val_loss'].to_numpy().min()
    face_loss_3 = face_history_3['val_loss'].to_numpy().min()
    face_loss_4 = face_history_4['val_loss'].to_numpy().min()
    face_loss_5 = face_history_5['val_loss'].to_numpy().min()
    face_loss_6 = face_history_6['val_loss'].to_numpy().min()
    face_loss_7 = face_history_7['val_loss'].to_numpy().min()
    face_loss_8 = face_history_8['val_loss'].to_numpy().min()
    face_loss_9 = face_history_9['val_loss'].to_numpy().min()
    face_loss_10 = face_history_10['val_loss'].to_numpy().min()
    face_loss_11 = face_history_11['val_loss'].to_numpy().min()


    # Beispiel-Daten
    accuracies = [age_accuracy_1, age_accuracy_2, age_accuracy_3, age_accuracy_4, age_accuracy_5, age_accuracy_6,
                  age_accuracy_7, age_accuracy_8, age_accuracy_9, age_accuracy_10, age_accuracy_11]
    losses = [age_loss_1, age_loss_2,age_loss_3,age_loss_4,age_loss_5,age_loss_6,age_loss_7,age_loss_8,age_loss_9,
              age_loss_10,age_loss_11]

    # Namen oder Labels für jeden Punkt
    parameter_labels = ['1', '2', '3', '4', '5' '6', '7', '8', '9', '10', '11']

    # Scatter Plot erstellen
    plt.scatter(losses,accuracies, c='blue', marker='o', label='Accuracy-Loss Paare')

    # Labels für die Achsen
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')

    # Jeden Punkt beschriften
    for i, label in enumerate(parameter_labels):
        plt.annotate(label, (losses[i], accuracies[i]))

    # Plot anzeigen
    plt.legend()
    plt.show()

#modelType = 'model12_classification'
#path_to_model = '/home/sarah/Deep-Learning/MS3/Model/{}'.format(modelType)
#plotHistory(path_to_model)

accuracy_loss_combination()
