# Datenauswahl

## Trainingsdaten

### UTKFace

- 23708 Gesichter
- aussortiert:
  - Bilder mit falschem Label
    Bsp: 1_0_2_20161219161843718, 1_0_3_20170104230640081, 1_0_4_20161221193041157, 5_0_1_20170117193745507, 36_0_0_20170113210318892
  - Bilder, die keine Gesichter zeigen
    Bsp: 1_0_0_20170109193052283, 54_0_0_20170117191419939, 80_0_2_20170111210646563, 86_1_2_20170105174652949

### Vergleichsdaten

- animals10: 26.179
  -butterfly: 2.112
  -cat: 1.668
  -chicken: 3.098
  -cow: 1.866
  -dog: 4.863
  -elefant: 1.446
  -horse: 2.623
  -sheep: 1.8220
  -spider: 4.821
  -squirrel: 1.862
- landscape: 4.319
- monkey: 1.098
- natural-images: 5.913
  -airplane: 727
  -car: 968
  -cat: 885
  -dog: 702
  -flower: 843
  -fruit: 1.000
  -motorbike: 788
  -person: 986

-aussortiert: Ordner person in natural-images,Bilder auf denen auch Personen sind:
-Bsp: dog_89, landscape_227, butterfly_487, cat_82 (animals10), chicken_1939, chicken_2822, chicken_2921,cow_150

### Datenset

Train_Test_Folder

- face 23.692
- noFace 23.692
