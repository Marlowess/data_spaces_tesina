# Tesina del corso Data Spaces
# Anno accademico 2018/2019
# Stefano Brilli, matricola s249914

import pandas as pd
import numpy as np

# Questa funzione legge i dati dal file elimina le righe che contengono valori non numerici e mappa l'output sui valori 0/1
# Se quality <= 5 --> 1
# Altrimenti quality --> 0
def read_data():
    path = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/Data Spaces/tesina/wine_quality/'
    df = pd.read_csv(path + 'winequality-red.csv', sep=';')  
    df = df[df.applymap(np.isreal).any(1)]
    df['quality'] = df['quality'].map(lambda x: 1 if x >= 6 else 0) 
    return df

df = read_data()
df.head(10)