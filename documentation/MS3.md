## Raw Data
Link: https://drive.google.com/file/d/1Z-lu2npV4vGvAYm1OiawZG05VRU13W8w/view?usp=sharing

Info zum Ordner UTKFace: Die Bilder sind wie folgt codiert: <br>
{alter}_{geschlecht}_face_{id}.jpg <br>
Alter: [0...116]<br>
Geschlecht: [0,1], wobei 0 = männlich und 1 = weiblich <br>

# Aufgabe: Multi-Task-Learning
## Option 1: Multi-Task-Learning als Klassifikationsproblem mit diskreten Altersklassen

### NoFace-Klasse

### Gender-Klasse
Geschelcht:  0 = Männlich, 1 = Weiblich

### Age-Klasse
Age Classe:
- age0: 1-2 Jahre
- age1: 3-9 Jahre
- age2: 10-20 Jahre
- age3: 21-27 Jahre
- age4: 28-45 Jahre
- age5: 46-65 Jahre
- age6: 66-116 Jahre

Erklärung der Age Classes:
age0: meistens keine Haare, kaum Zähne
age1: Gesichtszüge fangen an sich zu definieren
age2: Gesichtzüge werden erwachsener
age3: Erwachsene
age4: Erwachsene
age5: Erste Zeichen des Alterns
age6: graue Haare, eingefallene Haut

## Option 2: Multi-Task-Learning als Multi-Output-Regression
TODO: Xaver

