## Raw Data
Link: https://drive.google.com/file/d/1Z-lu2npV4vGvAYm1OiawZG05VRU13W8w/view?usp=sharing

Info zum Ordner UTKFace: Die Bilder sind wie folgt codiert: <br>
{alter}_{geschlecht}_face_{id}.jpg <br>
Alter: [1...116]<br>
Geschlecht: [0,1], wobei 0 = männlich und 1 = weiblich <br>

# Aufgabe: Multi-Task-Learning
## Option 1: Multi-Task-Learning als Klassifikationsproblem mit diskreten Altersklassen

### Daten erstellen:
MS3_rawData entspacken <br>
csv Datein mit preprocessing/preprocessing.py
- createFaceCSV_classification()
- createnoFaceCSV_classification()

### One-Hot-Encoding

gender = [0,1,2], wobei 0=männlich, 1=weiblich, 2=kein Geschlecht

face = [0,1], wobei 0=Gesicht, 1=kein Gesicht

age = [age0,age1,age2,age3,age4,age5,age6,age7], wobei
- age0: 1-2 Jahre
- age1: 3-9 Jahre
- age2: 10-20 Jahre
- age3: 21-27 Jahre
- age4: 28-45 Jahre
- age5: 46-65 Jahre
- age6: 66-116 Jahre
- age7: keine Zuordnung

Erklärung der Age Classes:
age0: meistens keine Haare, kaum Zähne
age1: Gesichtszüge fangen an sich zu definieren
age2: Gesichtzüge werden erwachsener
age3: Erwachsene
age4: Erwachsene
age5: Erste Zeichen des Alterns
age6: graue Haare, eingefallene Haut

### Classification Model
multiTaskLearning_classification.py ausführen

### Model auf deutschem Promi-Datenset testen
Promi-Datenset herunterladen: https://drive.google.com/file/d/1V8r50K1JF_DHNqq4mzetxjoxgnC4eyYm/view?usp=sharing
Entpacken und csv-Datei erstellen mit preprocessing/preprocessing.py
- promisToCSV_classification()

predict_classification.py ausführen

## Option 2: Multi-Task-Learning als Multi-Output-Regression

### Daten erstellen:
MS3_rawData entpacken <br>
csv Datein erstellen mit preprocessing/preprocessing.py:
- createFaceCSV_regression()
- createNoFaceCSV_regression()

### One-Hot-Encoding

gender = [0,1,2], wobei 0=männlich, 1=weiblich, 2=kein Geschlecht

face = [0,1], wobei 0=Gesicht, 1=kein Gesicht

age = [0..116], wobei 0=Alter kann nicht zugeordnet werden

### Regression Model
multiTaskLearning_regression.py ausführen

### Model auf deutschem Promi-Datenset testen
Promi-Datenset herunterladen: https://drive.google.com/file/d/1V8r50K1JF_DHNqq4mzetxjoxgnC4eyYm/view?usp=sharing
Entpacken und csv-Datei erstellen mit preprocessing/preprocessing.py
- promisToCSV_regression()
