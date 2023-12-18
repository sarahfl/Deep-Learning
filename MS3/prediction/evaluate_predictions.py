import numpy as np

AGE_RANGES = ["[1..2]", "[3..9]", "[10..20]", "[21..27]", "[28..45]", "[46..65]", "[66..116]", "No Age"]


def evaluate(predictions):
    predictions_age = predictions[0]
    predictions_gender = predictions[1]
    predictions_face = predictions[2]

    predicted_age_labels = np.argmax(predictions_age, axis=1) # returns the age if age is regression
    predicted_gender_labels = np.argmax(predictions_gender, axis=1)
    predicted_face_labels = np.argmax(predictions_face, axis=1)
    age_array = []
    if len(predictions_age[0]) > 1:
        for age in predicted_age_labels:
            age_array.append(AGE_RANGES[age])
    else:
        age_array = np.round(predictions_age.flatten())

    # revert gender classification
    gender_array = []
    for gender in predicted_gender_labels:
        ge = ''
        if gender == 0:
            ge = 'm'
        elif gender == 1:
            ge = 'f'
        else:
            ge = 'no'
        gender_array.append(ge)

    # revert face classification
    face_array = []
    for face in predicted_face_labels:
        fa = ''
        if face == 0:
            fa = 'yes'
        else:
            fa = 'no'
        face_array.append(fa)

    return age_array, gender_array, face_array
