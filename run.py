#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Analisi della qualità del vino Vinho Verde</h1>
# <h3 align="center">Stefano Brilli - Marzo 2019</h3>
# <hr><hr>

# <h2 align="center">Introduzione</h2>

# <img src="img/wine.jpg" width="600">

# In questo report saranno analizzati dati riguardanti il vino portoghese <b>Vinho Verde</b> nella sua variante rossa, allo scopo di predire se, in base ai valori osservati, si stia osservando un vino di bassa o alta qualità. Il dataset utilizzato è reperibile all'indirizzo http://archive.ics.uci.edu/ml/datasets/Wine+Quality.
# Al fine di costruire un classificatore binario, il dataset è stato modificato per rendere la variabile di output <b>(quality)</b> una quantità binaria. In particolare, le osservazioni aventi un output minore od uguale a 5 sono stati etichettati con il <b>valore 0 (bassa qualità)</b> mentre quelli aventi un valore qualitativo di 6 o superiore sono stati etichettati con il <b>valore 1 (alta qualità)</b>.<br>
# È chiaro come una mappatura di questo tipo ponga sia i vini con punteggio 6 che quelli con punteggio 10 al pari, nella categoria dei vini di alta qualità. D'altra parte, essa potrebbe permettere ai produttori di avere un iniziale responso della qualità del proprio vino, scartando in prima battuta quelli classificati come di bassa qualità ed andando successivamente ad analizzare nel dettaglio i restanti vini al fine di determinare una scala delle qualità.<br>
# I valori di input sono stati calcolati per mezzo di test fisico-chimici, mentre l'output è stato determinato tramite analisi sensoristiche.
# Gli attributi del dataset sono i seguenti:

# <ul>
#   <li>acidità fissa</li>
#   <li>acidità volatile</li>
#   <li>acido citrico</li>
#   <li>zucchero residuo</li>
#   <li>cloruri</li>
#   <li>anidride solforosa libera</li>
#   <li>anidride solforosa totale</li>
#   <li>densità</li>
#   <li>pH</li>
#   <li>solfati</li>
#   <li>alcool</li>
# </ul>

# <hr>
# <h3 align="center">Strumenti utilizzati per l'analisi</h3>
# L'intera analisi è stata condotta utilizzando il linguaggio <b>Python</b> e l'ambiente <b>Jupyter</b>.<br>
# Le librerie utilizzate sono:
# <ul>
#     <li><b>pandas</b> (<a href="https://pandas.pydata.org/">https://pandas.pydata.org/</a>), libreria software per la manipolazione e l'analisi dei dati</li>
#     <li><b>numpy</b> (<a href="http://www.numpy.org/">http://www.numpy.org/</a>), estensione open source del linguaggio di programmazione Python, che aggiunge supporto per vettori e matrici multidimensionali e di grandi dimensioni e con funzioni matematiche di alto livello con cui operare</li>
#     <li><b>plotly</b> (<a href="https://plot.ly/feed/">https://plot.ly/feed/</a>), insieme di strumenti per l'analisi e la visualizzazione dei dati</li>
#     <li><b>sklearn</b> (<a href="https://scikit-learn.org/stable/">https://scikit-learn.org/stable/</a>), libreria open source di apprendimento automatico per il linguaggio di programmazione Python</li>
# </ul>    

# <hr><hr>
# <h2 align="center">Esplorazione e visualizzazione dei dati</h2>

# Il dataset contiene <b>1599 righe</b> e <b>12 colonne</b> (11 di input + 1 di output). Il dataset non è stato alterato nel numero di osservazioni in quanto non risultavano valori mancanti.

# In[249]:


# Tesina del corso Data Spaces
# Anno accademico 2018/2019
# Stefano Brilli, matricola s249914

import numpy as np
import pandas as pd


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA as sklearnPCA

import copy

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
from plotly.offline import plot, iplot
import plotly.plotly as py

from IPython.display import Math
import sys

from sklearn.svm import LinearSVC
from warnings import simplefilter

import math

simplefilter(action='ignore', category=FutureWarning)

np.set_printoptions(threshold=sys.maxsize)
# Questa funzione legge i dati dal file elimina le righe che contengono valori non numerici e mappa l'output sui valori 0/1
# Se quality <= 5 --> 0
# Altrimenti quality --> 1
def read_data():
    path = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/Data Spaces/tesina/wine_quality/'
    df = pd.read_csv(path + 'winequality-red.csv', sep=';')  
    df = df[df.applymap(np.isreal).any(1)]
    df['quality'] = df['quality'].map(lambda x: 1 if x >= 6 else 0) 
    return df


# In[2]:


df = read_data()
df.head(10)


# Osservando soltanto queste prime dieci righe possiamo notare uno sbilanciamento tra i vini di bassa qualità e quelli di alta qualità. Ovviamente il campione non può essere preso come rappresentativo dell'intero dataset. Nel grafico seguente viene quindi mostrata la distribuzione delle due etichette di output.

# In[3]:


df.describe()


# In[4]:


colors = ['#2ca02c', '#d62728']
quality_values = {0: "Bassa", 1: "Alta"}


# In[5]:


y = df["quality"].value_counts()

data = [go.Bar(x=[quality_values[x] for x in y.index], y=y.values, marker = dict(color = colors[:len(y.index)]))]
layout = go.Layout(
    title='Distribuzione della qualità',
    autosize=False,
    width=400,
    height=400,
    yaxis=dict(
        title='Numero di osservazioni',
    ),
    xaxis=dict(
        title='Qualità'
    ),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic-bar3')


# Come è possibile osservare i dati sono distrubuiti in maniera quasi bilanciata relativamente al valore di output e questo potrebbe significare che il modello di predizione sarà in grado di discriminare bene tra le due classi nella fase di classificazione di nuovi dati. <br>Nel dettaglio, i dati si distribuiscono come segue:
#   
# <ul>  
#     <li><b>Alta qualità</b> (855 su 1599) = <b>53.47%</b></li>
#     <li><b>Bassa qualità</b> (744 su 1599) = <b>46.53%</b></li>
# </ul>

# <hr>
# <h3 align="center">Distribuzione dei valori delle feature</h3>

# Utilizzando i box plot è possibile osservare in che modo i valori delle feature sono distribuiti. Essi ci permettono di leggerne valore medio, massimo, minimo e rilevare eventuali outlier, ossia dati che non seguono il trend del gruppo a cui appartengono. Se presenti in numero abbastanza elevato, tali valori possono in alcuni casi rendere più complicata l'analisi, perchè il modello apprenderà pattern che si allontanano dal reale andamento dei dati in analisi.
# I valori sono stati normalizzati al fine di apprezzare meglio la visualizzazione, in quanto alcune feature presentavano valori estremamente alti rendendo il grafico schiacciato verso il basso.

# In[6]:


# Normalizzazione dei dati (tra 0 e 1)
X = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
df_norm = pd.DataFrame(x_scaled)

# Costruzione delle strutture per rappresentare i box plot
data = []
for i in range(len(df.columns) - 1):
    data.append(go.Box(
        name = df.columns[i],
        y = df_norm[i].tolist()        
    ))
df_norm.columns = df.columns    
fig = go.Figure(data=data)
iplot(fig) 


# È evidente come alcune delle feature si distribuiscano molto lungo l'intervallo dei valori e questo potrebbe comportare una minore accuratezza dei classificatori. Per tale motivo si è deciso di eliminarle dal dataset. Esse sono <b>residual sugar</b>, <b>chlorides</b> e <b>sulphates</b>.

# In[7]:


# Rimozione delle feature che presentano valori troppo distribuiti, in accordo con il box plot precedente
df_norm = df_norm.drop(columns=['residual sugar', 'chlorides', 'sulphates'])


# <hr>
# <h3 align="center">Correlazione tra le feature</h3>

# Il grafico sottostante riporta la correlazione che intercorre tre la feature. Una correlazione positiva indica che quando la prima feature cresce anche la seconda cresce. Una correlazione negativa indica una tendenza inversa all'aumento: quando la prima aumenta la seconda tende a diminuire.

# In[8]:


df_ = copy.copy(df_norm).drop(columns=['quality'])
covariance_matrix = df_.corr()
covariance_matrix = covariance_matrix.values
x_ = []
y_ = []
z_ = []

for i in range(len(df_.columns)):
    z_.append(covariance_matrix[i,])
    x_.append(df_norm.columns[i])
    y_.append(df_norm.columns[i])

trace = go.Heatmap(z=z_,
                   x=x_,
                   y=y_)    
figure = go.Figure(data=[trace])
iplot(figure)


# <hr><hr>
# <h2 align="center">Analisi delle componenti principali (PCA)</h2>

# L'analisi delle componenti principali (PCA, dall'inglese <b>Principal Components Analysis</b>) è una tecnica che nasce dalla necessità di migliorare le prestazioni degli algoritmi di analisi dei dati a fronte di un enorme incremento della numerosità di essi a disposizione degli analisti. Lo scopo della PCA è dunque quello di rilevare una possibile correlazione tra le diverse variabili di un dataset tale per cui la maggioranza della variabilità dell'informazione dipenda soltanto da esse. Riproiettando i dati lungo gli assi rilevati e scartando le direzioni povere di variabilità è possibile ridurre la dimensionalità del dataset, migliorando le prestazioni ed ottimizzando le risorse.
# Per il calcolo delle componenti principali l'approccio è il seguente:
# <ul>  
#     <li>i dati vengono standardizzati</li>
#     <li>si calcolano gli autovalori e gli autovettori dalla matrice di covarianza o da quella di correlazione, oppure si calcola la decomposizione a valore singolo</li>
#     <li>si ordinano gli autovalori in ordine decrescente e si calcolano gli autovettori associati ad essi</li>
#     <li>si costruisce la matrice W dei dati riproiettati utilizzando gli autovettori selezionati</li>
#     <li>si trasforma il dataset originale tramite la matrice W per ottenere un sottospazio Y delle feature di dimensione pari al numero di autovettori utilizzati</li>
# </ul>
# 
# Nel contesto di questo report sono state utilizzate le funzioni messe a disposizioni da Sklearn per il calcolo della PCA, le quali utilizzano il metodo della decomposizione a valore singolo (SVD) per trovare le componenti principali.

# In[9]:


def doPCA(X_std, components, print_flag):    
    X_data = copy.copy(X_std)  
    
    # Rimappo le etichette per mostrare la legenda sul grafico finale
    X_data['quality'] = X_data['quality'].map(lambda x: 'Alta qualità' if x == 1 else 'Bassa qualità')   
    
    # Copio le etichette nel vettore y e le rimuovo da X
    y = X_data['quality']
    X_data = X_data.drop(columns=['quality'])
    
    # Calcolo le componenti principali
    sklearn_pca = sklearnPCA(n_components = components)
    Y_sklearn = sklearn_pca.fit_transform(X_data) 
    data = []
    colors = { 'red': '#2ca02c', 'green': '#d62728'}
    
    # Per ogni elemento costruisco una traccia dello scatter plot
    for name, col in zip(('Alta qualità', 'Bassa qualità'), colors.values()):
        trace = dict(
            type='scatter',
            x = Y_sklearn[y==name,0],
            y = Y_sklearn[y==name,1],
            mode = 'markers',
            name = name,
            marker = dict(
                color = col,
                size = 12,
                line = dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8)
        )
        data.append(trace)

    layout = dict(
            xaxis=dict(title='PC1', showline=True),
            yaxis=dict(title='PC2', showline=True)
    )
    if(print_flag):
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)
            
    # Ritorno i dati riproiettati e l'oggetto PCA
    return Y_sklearn, sklearn_pca


# Di seguito sono raffigurati tutte le osservazioni del dataset riproiettate lungo le principali due componenti. Questo è fatto a solo scopo di mostrare la distribuzione dei punti, perchè come si vedrà poco sotto, sarà utilizzato un numero maggiore di componenti per l'analisi.

# In[10]:


doPCA(df_norm, 2, True)
my_pca_data, my_pca_obj = doPCA(df_norm, 4, False)
my_total_pca_data, my_total_pca_obj = doPCA(df_norm, 8, False)


# Il grafico successivo mostra la quantità di varianza raccolta da ciascuna componente, dove possiamo notare come siano sufficienti le prime quattro componenti per spiegare più del 85% della varianza totale.

# In[11]:


# Funzione che serve a rappresentare graficamente la quantità di varianza che ogni componente
# principale è in grado di raccogliere
def print_explained_variance(explained_variance):    
    cumulative_variance = []
    cumulative_variance.append(explained_variance[0])
    
    # Ogni cella dell'array contiene tutta la varianza raccolta finora + quella della relativa cella
    for i in range(1, 8):
        old = cumulative_variance[i-1]
        new = explained_variance[i]
        cumulative_variance.append(old + new)
    
    # Una traccia per la varianza di ogni componente, l'altra per la varianza cumulativa
    data = [go.Bar(
            x=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'],
            y=[explained_variance[0], explained_variance[1], explained_variance[2], 
               explained_variance[3], explained_variance[4], explained_variance[5],
               explained_variance[6], explained_variance[7]],
            name="Varianza spiegata"
    ), go.Bar(
            x=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'],
            y=cumulative_variance,
            name="Varianza cumulativa"
    )]
    layout = go.Layout(
        title='Varianza spiegata e cumulativa',
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig) 


# In[12]:


print_explained_variance(my_total_pca_obj.explained_variance_ratio_)


# <hr><hr>
# <h2 align="center">Overfitting e valutazione delle prestazioni del modello</h2>

# Costruire un modello utilizzando un unico set di training è un errore che va evitato, perchè esso apprenderà in maniera pressochè perfetta tutto ciò che riguarda tale set ma le sue prestazioni su nuovi dati potrebbero essere disastrose. Questa condizione è chiamata <b>overfitting</b> ed andrebbe evitata se si vuole costruire un buon classificatore. Per fare ciò si possono utilizzare diverse tecniche atte ad allenare il modello utilizzando diversi set di training, in modo tale che il modello non si "abitui" ad uno specifico set ma sia in grado di generalizzare il più possibile con lo scopo di classificare meglio i dati di test, quelli che non sono stati usati per l'apprendimento.<br>
# In questa analisi sarà utilizzata la tecnica di validazione incrociata chiamata <b>k-Fold</b> che svolge le seguenti operazioni:
# <ul>
#     <li>il dataset viene suddiviso in k diversi subset</li>
#     <li>Per ciascuno dei k subset:</li>
#     <ul>
#         <li>il modello viene allenato utilizzando i k-1 rimanenti subset</li>
#         <li>il modello finale viene testato sulla rimanente porzione di dati (viene cioè calcolata l'accuratezza)</li>
#     </ul>
# </ul>
# L'accuratezza complessiva del modello è data dalla media delle accuratezze ottenute per ciascuno dei passi descritti sopra.
# <img src="img/grid_search_cross_validation.png" width="600">

# <hr><hr>
# <h2 align="center">Validazione del modello</h2>

# Per ciascuno degli algoritmi di classificazione che useremo sarà quantificata la qualità della previsione utilizzando la metrica <b>F1-score</b>, definita come $$ F1_{score} = 2 \frac{\text{precisione} \times \text{sensibilità}}{\text{precisione} + \text{sensibilità}} $$
# 
# $$ \text{precisione} = \frac{T_{P}}{T_{P} + F_{P}} = \frac{Veri\ positivi}{Totale\ positivi\ previsti}$$
# $$ \text{sensibilità} = \frac{T_{P}}{T_{P} + F_{N}} = \frac{Veri\ positivi}{Totale\ positivi\ reali}$$
# 
# <ul>
#     <li><b>TP</b>: Vero positivo (risultato corretto)</li>
#     <li><b>TN</b>: Vero negativo (corretta assenza di risultato)</li>
#     <li><b>FP</b>: Falso positivo (risultato inatteso)</li>
#     <li><b>FN</b>: Falso negativo (risultato mancante)</li>
# </ul><br>
# 
# La <b>precisione</b> è il rapporto tra le osservazioni predette in maniera corretta rispetto alle osservazioni totali predette come positive. Un'alta precisione significa che il modello ha predetto poche volte la classe positiva quando la classe giusta era quella negativa (basso valore di falsi positivi).<br><br>
# La <b>sensibilità</b> è il rapporto tra le osservazioni correttamente classificate come positive rispetto a tutte le osservazioni che fanno parte di tale categoria. Essa rappresenta intuitivamente l'abilità del classificatore nel trovare tutte le osservazioni della classe positiva.<br><br>
# Il valore <b>F1</b> calcola la media pesata tra precisione e sensibilità. Quindi esso tiene in considerazione sia i falsi positivi che che i falsi negativi.<br><br>
# In un'analisi di questo tipo è bene decidere e quindi valutare quale peso dare agli errori. In altre parole dobbiamo chiederci se sia meglio ridurre al minimo il numero di esiti falsi positivi (e dunque avere alta precisione) oppure quello di falsi negativi (maggiore sensibilità). <br>
# In questo caso potrebbe essere sensato <b>puntare sulla precisione</b>, in quanto distribuire un vino classificato come di alta qualità e poi ricevere lamentele da parte dei venditori (danno d'immagine) è sicuramente peggio che buttare del vino che si pensava essere di scarsa qualità quando era in realtà un buon vino (perlomeno per le grandi aziende, dove la quantità di prodotto è sicuramente alta). <br>
# 
# 
# Sarà inoltre mostrata per ogni algoritmo la <b>curva ROC</b> e l'area sottesa dalla curva (<b>AUC</b>).<br>
# La ROC è una curva di probabilità e AUC rappresenta il grado di misura della <b>separabilità</b>. In altre parole ci dice quanto il modello è abile a distinguere (in termini di <b>probabilità</b>) le due classi. Più è alto il valore di AUC e meglio il modello sarà in grado di prevedere 0 quando è 0 ed 1 quando è 1. In relazione a questa analisi, più è alto il valore di AUC e meglio il modello sarà in grado di classificare un vino come di alta qualità quando lo è effettivamente, e viceversa.
# La curva ROC è rappresentata con i valori di <b>TPR (True Positive Rate)</b> sull'asse X e con i valori di <b>FPR (False Positive Rate)</b> sull'asse Y.<br>
# Un modello eccellente ha un valore di AUC pari a 1, il che significa che ha un ottimo grado di separabilità. Viceversa un modello con AUC vicino allo zero non è assolutamente in grado di separare le due classi, e anzi classificherà in maniera inversa rispetto alla realtà. Infine quando AUC = 0.5 significa che il modello non ha la capacità di separare i dati delle due classi.<br>
# Sotto sono mostrati degli esempi di curve ROC in base al valore di AUC.
# 
# <table>
#     <tr>
#         <td>
#             <img src="img/roc_auc_1.png" width="300">
#         </td>
#         <td>
#             <img src="img/roc_auc_1_1.png" width="150">
#         </td>
#         <td>
#             <img src="img/roc_auc_0_7.png" width="300">
#         </td>
#         <td>
#             <img src="img/roc_auc_0_7_1.png" width="150">
#         </td>
#     </tr>
#     <tr>
#         <td>
#             <img src="img/roc_auc_0_5.png" width="300">
#         </td>
#         <td>
#             <img src="img/roc_auc_0_5_1.png" width="150">
#         </td>
#         <td>
#             <img src="img/roc_auc_0.png" width="300">
#         </td>
#         <td>
#             <img src="img/roc_auc_0_1.png" width="150">
#         </td>
#     </tr>
# </table>
# 
# 
# 
# 

# <hr><hr>
# <h2 align="center">Classificazione</h2>

# In questo capitolo saranno utilizzati diversi algoritmi di classificazione e, per ciascuno di essi, sarà fornita una adeguata spiegazione teorica, così che anche il lettore non esperto di questo campo possa capire e cogliere le motivazione che stanno dietro allo studio qui presentato.<br>
# Sara utilizzato il dataset ridotto tramite PCA, formato dalle prime quattro componenti (che spiegano una varianza del 86% che possiamo ritenere sufficiente per una buona analisi).<br>
# Gli algoritmi di classificazione che saranno utilizzati sono:
# <ul>
#     <li><b>K-Nearest Neighbors (KNN)</b></li>
#     <li><b>Support Vector Machine (SVM)</b></li>
#     <li><b>Random Forest</b></li>
#     <li><b>Regressione Logistica</b></li>
# </ul>

# In[290]:


# Questa classe gestisce e mantiene al suo interno tutte le informazioni sui classificatori e sui risultati, in modo
# da avere un unico contenitore di dati, grafici e risultati.
# Per eseguire un confronto sensato tra i classificatori, il dataset sarà
class Multiclassifier:
    def __init__(self, df, df_norm):
        # Etichette
        self.y = df['quality'].values

        # Elimino la colonna delle etichette dal dataframe
        self.X = copy.copy(df_norm).drop(columns=['quality']).values
        
        # Creo il dizionario dei colori
        self.colors_dict =	{
          "grey": "rgba(233, 233, 233 ,.8)",
          "orange": "rgba(255, 127, 14,.8)"
        }
        
        # Costruisco un oggetto K-Fold che userò nei cicli for dopo. Per non favorire nessuno degli algoritmi lo creo
        # qui, così che tutti i modelli saranno costruiti con la stessa suddivisione
        self.kf = KFold(n_splits=5, shuffle=True)
        
        
    
    # Metodo per costruire le tracce di curve ROC
    def build_roc_trace(self, fpr, tpr, color):                        
        # Singola traccia di curva ROC
        trace = go.Scatter(x = fpr, y = tpr, mode = 'lines', marker = dict(
            size = 10, color = self.colors_dict[color],
            line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)')))
                
        # Linea tratteggiata
        trace_dash_line = go.Scatter(x=[0, 1], y=[0, 1], 
            mode='lines', 
            line=dict(color='navy', width=2, dash='dash'),
            showlegend=False)
        return trace, trace_dash_line
    
    # Metodo per plottare le curve ROC costruite
    def plot_roc_curves(self, roc_traces_list):
        layout_roc = dict(
                title="Curve ROC per tutti i fold e curva ROC globale",
                xaxis = dict(title = "FPR"),
                yaxis = dict(title = "TPR"),
                showlegend=False)
        fig_roc = go.Figure(data=roc_traces_list, layout=layout_roc)
        iplot(fig_roc)         
        
    
    # Questo metodo itera lungo tutti i fold in cui il dataset è stato suddiviso restituendo i risultati
    # dentro ad array
    def k_fold_loop(self, clf, roc_traces, tprs, fpr_mean, auc_area, f1_scores, precision_scores, recall_scores):        
        f1_scores = []        
        precision_scores = []        
        recall_scores = []  
        roc_tpr = []
        roc_fpr = []        

        # Itero su ciascun fold fornito dall'algoritmo K-Fold
        for train_index, test_index in self.kf.split(self.X):            
            clf.fit(self.X[train_index], self.y[train_index])

            # Costruisco gli array parziali per f1-Score
            y_test_predict = clf.predict(self.X[test_index])            
            precision_scores = np.append(precision_scores, precision_score(self.y[test_index], y_test_predict))
            recall_scores = np.append(recall_scores, recall_score(self.y[test_index], y_test_predict))
            f1_scores = np.append(f1_scores, f1_score(self.y[test_index], y_test_predict, labels=np.unique(y_test_predict)))   

            # Per ogni fold costruisco una curva ROC, inserendo una traccia nell'array. Le plotterò alla fine
            y_score = clf.predict_proba(self.X[test_index])
            fpr, tpr, _ = roc_curve(self.y[test_index], y_score[:, 1])
            auc = roc_auc_score(self.y[test_index], y_score[:, 1])
            auc_area = np.append(auc_area, auc)
            trace, _ = self.build_roc_trace(fpr, tpr, "grey")

            # Appendo i risultati parziali all'array globale. Serve per calcolare la media per ogni array
            # contenente i risultati ritornati dalla funzione roc_curve
            roc_traces = np.append(roc_traces, trace)                               
            roc_tpr = np.append(roc_tpr, tpr)
            tprs.append(interp(fpr_mean, fpr, tpr))
            tprs[-1][0] = 0.0
        return roc_traces, tprs, fpr_mean, auc_area, f1_scores, precision_scores, recall_scores
                
                
        
    # Costruisco il grafico del F1-score, passando l'array di valori da mostrare sull'asse x 
    # ed i tre array contenenti precisione, sensibilità, f1-score
    def plot_f1_trace(self, x_array, precisione, recall, f1, x_axis, y_axis):
        trace1 = go.Scatter(
            x = x_array,
            y = precisione,
            name = "Precisione"
        )

        trace2 = go.Scatter(
            x = x_array,
            y = recall,
            name = "Sensibilità"
        )

        trace3 = go.Scatter(
            x = x_array,
            y = f1,
            name = "F1-Score"
        )

        layout = dict(
                  title="Andamento di F1-score, precisione e sensibilità",
                  yaxis = y_axis,
                  xaxis = x_axis,
                  barmode='group'
                 )

        data = [trace1, trace2, trace3]

        fig = go.Figure(data=data, layout=layout)
        iplot(fig) 
        
    # Metodo che trova il minimo e il massimo di questi vettori per poter disegnare un grafico che racchiuda 
    # tutti i dati
    def min_max_finder(self, array_1, array_2, array_3):
        array = []
        array = np.append(array,array_1)
        array = np.append(array,array_2)
        array = np.append(array,array_3)
        
        min_y = np.amin(array) - 0.05
        max_y = np.amax(array) + 0.05
        
        return min_y, max_y
    
    # Metodo per disegnare il valore di F1-Score tra due parametri di cui facciamo tuning
    def plotHeatMapSVM(self, c_array, gamma_array, accuracy_array):
        trace = go.Heatmap(z=accuracy_array,
                           x=gamma_array,
                           y=c_array)   
        layout_svm_rbf = dict(
              title="Andamento del F1-Score al variare di C e Gamma",
              yaxis = dict(zeroline = False, title="C", type="log", tickmode="linear", tick0=-3, dtick=1),
              xaxis = dict(title="Gamma", type="log", tickmode="linear", tick0=math.log10(gamma_array[0]), dtick="log_10(2)")
             )

        figure = go.Figure(data=[trace], layout=layout_svm_rbf)
        iplot(figure)
    
    # Metodo per la l'analisi del dataset usando l'algoritmo KNN
    def KNN_analysis(self, param_range):        

        # Inizializzazione degli array per salvare i dati temporanei e quelli per disegnare i grafici
        # Array dei valori di K
        k_values = []

        # Array per salvare i valori di F1, precisione e recall. Mi servono per i plot
        f1 = []
        precisione = []
        recall = []  

        # Array utili per mantenere le informazioni per la curva ROC finale
        roc_traces = []
        tpr_mean = []  
        tprs = []
        auc_area = [] 
        fpr_mean = np.linspace(0, 1, 100)
        
        for k in param_range:    
            # Costruisco un classificatore ad ogni iterazione del ciclo
            clf = KNeighborsClassifier(n_neighbors=k)        

            # Array simili a quelli dichiarati fuori dai cicli for, ma utilizzati per i risultati
            # parziali per ciascun valore di k
            f1_scores = []        
            precision_scores = []        
            recall_scores = []  
            roc_tpr = []
            roc_fpr = []        

            # Itero su ciascun fold fornito dall'algoritmo K-Fold
            roc_traces, tprs, fpr_mean, auc_area, f1_scores, precision_scores,                 recall_scores = self.k_fold_loop(clf, roc_traces, tprs, fpr_mean, auc_area, f1_scores,                                                  precision_scores, recall_scores)
            
            # Calcolo i valori totali relativi a questa k e li appendo all'array globale dei risultati
            k_values = np.append(k_values, k)
            f1 = np.append(f1, f1_scores.mean())
            precisione = np.append(precisione, precision_scores.mean())
            recall = np.append(recall, recall_scores.mean())    

        # Calcolo i valori y della curva ROC globale
        #tpr_mean = np.mean(tprs, axis=0)
        max_f1_index = np.argmax(f1)
        max_precisione_index = np.argmax(precisione)
        tpr_mean = tprs[max_f1_index]
        tpr_mean[-1] = 1.0                        

        # Calcolo l'area AUC media
        auc = auc_area[max_f1_index]

        # Costruisco la curva ROC generale passandogli gli array globali costruiti finora        
        trace_mean_roc, trace_dash_line = self.build_roc_trace(fpr_mean, tpr_mean, "orange")

        # Infine appendo questa traccia insieme alle altre
        roc_traces = np.append(roc_traces, trace_mean_roc)
        roc_traces = np.append(roc_traces, trace_dash_line)
        
        # Traccio il grafico di precisione, sensibilità e f1-score
        min_y, max_y = self.min_max_finder(f1, precisione, recall)
        self.plot_f1_trace(k_values, precisione, recall, f1, dict(title="K"),                            dict(zeroline=False, title="Score", range=[min_y,max_y]))
        
        self.plot_roc_curves(roc_traces.tolist())
        
        auc_string = "AUC (associato a classificatore con migliore F1-Score) = %0.2f" % auc
        f1_string = "Risultati classificatore con obiettivo F1-Score MAX\n\t{ F1-Score: %0.3f, {Precisione: %0.3f, Sensibilità: %0.3f}, K = %d}"             % (f1[max_f1_index], precisione[max_f1_index], recall[max_f1_index], k_values[max_f1_index])
        precision_string = "Risultati classificatore con obiettivo Precisione MAX\n\t{ Precisione: %0.3f, {F1-Score: %0.3f, Sensibilità: %0.3f}, K = %d}"             % (precisione[max_precisione_index], f1[max_precisione_index],                recall[max_precisione_index],k_values[max_precisione_index])
        
        print(auc_string)
        print(f1_string)
        print(precision_string)
        
        # Salvo la curva ROC e il valore di AUC per le comparazioni finali
        self.knn_roc_curve = trace_mean_roc
        self.knn_auc = auc
    
    # Metodo per la l'analisi del dataset usando l'algoritmo SVM con kernel lineare
    def SVM_linear_analysis(self, param_range):
        # Costruisco un oggetto K-Fold che userò nei cicli for successivamente
        #kf = KFold(n_splits=5, shuffle=True)

        # Inizializzazione degli array per salvare i dati temporanei e quelli per disegnare i grafici
        # Array dei valori di c
        c_values = []

        # Array per salvare i valori di F1, precisione e recall. Mi servono per i plot
        f1 = []
        precisione = []
        recall = []  

        # Array utili per mantenere le informazioni per la curva ROC finale
        roc_traces = []
        tpr_mean = []  
        tprs = []
        auc_area = [] 
        fpr_mean = np.linspace(0, 1, 100)
        
        for c in param_range:    
            # Costruisco un classificatore ad ogni iterazione del ciclo
            clf = svm.SVC(C=c, kernel='linear', gamma='auto', probability=True)   
            #lf = svm.SVC(probability=True)

            # Array simili a quelli dichiarati fuori dai cicli for, ma utilizzati per i risultati
            # parziali per ciascun valore di k
            f1_scores = []        
            precision_scores = []        
            recall_scores = []  
            roc_tpr = []
            roc_fpr = []        

            # Itero su ciascun fold fornito dall'algoritmo K-Fold
            roc_traces, tprs, fpr_mean, auc_area, f1_scores, precision_scores,                 recall_scores = self.k_fold_loop(clf, roc_traces, tprs, fpr_mean, auc_area, f1_scores,                                                  precision_scores, recall_scores)
            
            # Calcolo i valori totali relativi a questa k e li appendo all'array globale dei risultati
            c_values = np.append(c_values, c)
            f1 = np.append(f1, f1_scores.mean())
            precisione = np.append(precisione, precision_scores.mean())
            recall = np.append(recall, recall_scores.mean())    

        # Calcolo i valori y della curva ROC globale
        max_f1_index = np.argmax(f1)
        max_precisione_index = np.argmax(precisione)
        tpr_mean = tprs[max_f1_index]
        tpr_mean[-1] = 1.0                        

        # Calcolo l'area AUC media
        auc = auc_area[max_f1_index]

        # Costruisco la curva ROC generale passandogli gli array globali costruiti finora        
        trace_mean_roc, trace_dash_line = self.build_roc_trace(fpr_mean, tpr_mean, "orange")

        # Infine appendo questa traccia insieme alle altre
        roc_traces = np.append(roc_traces, trace_mean_roc)
        roc_traces = np.append(roc_traces, trace_dash_line)
        
        # Traccio il grafico di precisione, sensibilità e f1-score
        min_y, max_y = self.min_max_finder(f1, precisione, recall)
        self.plot_f1_trace(c_values, precisione, recall, f1, dict(title="C", type="log",                           tickmode="linear", tick0=-3, dtick=1),                          dict(zeroline = False, title="Score", range = [min_y,max_y]))
        
        self.plot_roc_curves(roc_traces.tolist())
        
        auc_string = "AUC (associato a classificatore con migliore F1-Score) = %0.2f" % auc
        f1_string = "Risultati classificatore con obiettivo F1-Score MAX\n\t{ F1-Score: %0.3f, {Precisione: %0.3f, Sensibilità: %0.3f}, C = %d}"             % (f1[max_f1_index], precisione[max_f1_index], recall[max_f1_index], c_values[max_f1_index])
        precision_string = "Risultati classificatore con obiettivo Precisione MAX\n\t{ Precisione: %0.3f, {F1-Score: %0.3f, Sensibilità: %0.3f}, C = %d}"             % (precisione[max_precisione_index], f1[max_precisione_index],                recall[max_precisione_index],c_values[max_precisione_index])
        
        print(auc_string)
        print(f1_string)
        print(precision_string)
                
        # Salvo la curva ROC e il valore di AUC per le comparazioni finali
        self.svm_linear_roc_curve = trace_mean_roc
        self.svm_linear_auc = auc

        
    # Metodo utile alla costruzione del modello tramite utilizzo del kernel rbf 
    def SVM_rbf_analysis(self, c_range, gamma_range):
        # Costruisco un oggetto K-Fold che userò nei cicli for successivamente
        #f = KFold(n_splits=5, shuffle=True)

        # Inizializzazione degli array per salvare i dati temporanei e quelli per disegnare i grafici
        # Array dei valori di c e gamma
        c_values = []
        gamma_values = []

        # Array per salvare i valori di F1, precisione e recall. Mi servono per i plot
        f1 = []
        precisione = []
        recall = []  

        # Array utili per mantenere le informazioni per la curva ROC finale
        roc_traces = []
        tpr_mean = []  
        tprs = []
        auc_area = [] 
        fpr_mean = np.linspace(0, 1, 100)
        
        for c in c_range:    
            for gamma_ in gamma_range:
                # Costruisco un classificatore ad ogni iterazione del ciclo
                clf = svm.SVC(C=c, kernel='rbf', gamma=gamma_, probability=True)   
                #lf = svm.SVC(probability=True)

                # Array simili a quelli dichiarati fuori dai cicli for, ma utilizzati per i risultati
                # parziali per ciascun valore di k
                f1_scores = []        
                precision_scores = []        
                recall_scores = []  
                roc_tpr = []
                roc_fpr = []        

                # Itero su ciascun fold fornito dall'algoritmo K-Fold
                roc_traces, tprs, fpr_mean, auc_area, f1_scores, precision_scores,                     recall_scores = self.k_fold_loop(clf, roc_traces, tprs, fpr_mean, auc_area, f1_scores,                                                      precision_scores, recall_scores)

                # Calcolo i valori totali relativi a questa k e li appendo all'array globale dei risultati
                c_values = np.append(c_values, c)
                gamma_values = np.append(gamma_values, gamma_)
                f1 = np.append(f1, f1_scores.mean())
                precisione = np.append(precisione, precision_scores.mean())
                recall = np.append(recall, recall_scores.mean()) 

        # Calcolo i valori y della curva ROC globale
        max_f1_index = np.argmax(f1)
        max_precisione_index = np.argmax(precisione)
        tpr_mean = tprs[max_f1_index]
        tpr_mean[-1] = 1.0

        # Calcolo l'area AUC media
        auc = auc_area[max_f1_index]

        # Costruisco la curva ROC generale passandogli gli array globali costruiti finora        
        trace_mean_roc, trace_dash_line = self.build_roc_trace(fpr_mean, tpr_mean, "orange")

        # Infine appendo questa traccia insieme alle altre
        roc_traces = np.append(roc_traces, trace_mean_roc)
        roc_traces = np.append(roc_traces, trace_dash_line)
        
        # Plotto la heatmap che mostra il valore di F1-score al variare dei parametri C e Gamma
        self.plotHeatMapSVM(c_values, gamma_values, f1)
        
        # Plotto la curva ROC generale e tutte le altre
        self.plot_roc_curves(roc_traces.tolist())
        
        auc_string = "AUC (associato a classificatore con migliore F1-Score) = %0.2f" % auc
        f1_string = "Risultati classificatore con obiettivo F1-Score MAX\n\t{ F1-Score: %0.3f, {Precisione: %0.3f, Sensibilità: %0.3f}, C = %0.3f, Gamma = %0.3f}"             % (f1[max_f1_index], precisione[max_f1_index], recall[max_f1_index], c_values[max_f1_index], gamma_values[max_f1_index])
        precision_string = "Risultati classificatore con obiettivo Precisione MAX\n\t{ Precisione: %0.3f, {F1-Score: %0.3f, Sensibilità: %0.3f}, C = %0.3f, Gamma = %0.3f}"             % (precisione[max_precisione_index], f1[max_precisione_index],                recall[max_precisione_index],c_values[max_precisione_index], gamma_values[max_precisione_index])
        
        print(auc_string)
        print(f1_string)
        print(precision_string)            
        
        # Salvo la curva ROC e il valore di AUC per le comparazioni finali
        self.svm_rbf_roc_curve = trace_mean_roc
        self.svm_rbf_auc = auc


# In[293]:


my_multiclassifier = Multiclassifier(df, df_norm)


# <hr>
# <h3 align="center">K-Nearest Neighbors (KNN)</h3>

# Il KNN è un algoritmo utilizzato per costruire e classificare pattern di oggetti basandosi sulle caratteristiche di essi. Tra gli algoritmi utilizzati nel Machine Learning è quello più semplice da implementare e da comprendere ed il funzionamento è il seguente: i voti dei k vicini del campione determinano il gruppo a cui assegnare l'oggetto in esame. Per evitare situazioni di parità è bene scegliere un valore k che non sia un divisore del numero di gruppi distinti presenti nel dataset. Per k = 1 il campione viene assegnato al gruppo dell'oggetto a lui più vicino.
# Per scegliere il valore di k più adeguato è necessario valutare la tipologia di dati che si sta esaminando e la quantità di essi. Infatti, scegliere un valore di k elevato garantisce una migliore resistenza al rumore ma rende meno risconoscibili i margini di separazione.
# Saranno provati alcuni valori di K al fine di valuare quale fornisce il punteggio F1-score maggiore.

# Sono sotto riportati gli andamenti di precisione, recall, F1-score e le curve ROC al variare di K.
# Le curve ROC in grigio rappresentano il calcolo fatto per ciascuno dei fold e per ogni valore di K, mentre in blu è rappresentata la curva che approssima tutte le altre.

# In[294]:


# Uso il KNN testando tutti i valori di K compresi tra 4 e 40 con passo 2
my_multiclassifier.KNN_analysis(range(4, 40, 2))


# <hr>
# <h3 align="center">Support Vector Machine (SVM)</h3>

# L'SVM è un algoritmo supervisionato per classificare elementi appartenenti a due o più classi. L'obiettivo è quello di trovare l'iperpiano che sia in grado di separare i dati nel modo migliore. Questo significa assegnare, per ciascun punto, un etichetta in base a dove il punto si trovi rispetto all'iperpiano di separazione.
# Per trovare tale iperpiano abbiamo bisogno di risolvere un problema di ottimizzazione.
# 
# 
# <img src="img/svm_1.png" width="400">

# <hr>
# <h4 align="center">Il problema di ottimizzazione</h4>

# Se i dati sono perfettamente separabili nello spazio allora è dimostrato come sia possibile trovare l'iperpiano di separazione in un numero finito di passi. Ciò è garantito dal <b>teorema di convergenza del Perceptron</b>
# (<a href="http://www.cs.columbia.edu/~mcollins/courses/6998-2012/notes/perc.converge.pdf">http://www.cs.columbia.edu/~mcollins/courses/6998-2012/notes/perc.converge.pdf</a>).<br>
# A questo punto viene naturale chiedersi quale dei possibili iperpiani dobbiamo scegliere per separare i dati secondo le etichette. L'SVM per definizione ricerca l'iperpiano che massimizza la distanza tra i due gruppi di elementi (si veda l'immagine sopra).<br>
# $$ \text{ Formulazione primale: } \min\frac{1}{2}||\omega||^2 \text{ subject to: } y^{(t)}(\omega \cdot x^{(t)} + b) \geq k, \ t=1,...,n $$ <br>
# Si tratta di un <b>problema di ottimizzazione quadratico e con vincolo</b>. Esso può essere risolto utilizzando il metodo dei <b>moltiplicatori di Lagrange</b>, ed essendo un problema quadratico si tratta di trovare il minimo globale (unico) di una superficie paraboloidale.
# Calcolando le derivate parziali della formulazione primale otteniamo la formulazione alternativa del problema. Essa è definita come $$ \text{ Formulazione alternativa: } \max\text{-}\frac{1}{2}\sum_{ij}{\alpha_i\alpha_j y_i y_j \vec{x_i} \cdot \vec{x_j}} + \sum_i{\alpha_i} \text{ subject to: } \sum_i{\alpha_i y_i} = 0 \text{, } \alpha_i > 0 $$

# Formulando il problema in questo modo siamo in grado di calcolare la soluzione elimando la dipendenza dal vettore omega dei pesi e dal bias b. 
# L'unica dipendenza rimasta per la risoluzione è quella relativa ai coefficienti alfa.
# Questi coefficienti sono per lo più uguali a zero. Gli unici diversi da zero sono quelli relativi agli elementi presenti sui margini, che prendono il nome di <b>support vectors</b>, da cui il nome dell'algoritmo.

# Nel caso in cui i dati non siano separabili linearmente, come in questo caso, possiamo pensare di risolvere il problema di ottimizzazione permettendo ad alcuni elementi del dataset di essere dalla parte sbagliata del margine, o addirittura dalla parte sbagliata dell'iperpiano. Questo si formalizza aggiungengo un ulteriore vincolo alle formulazioni precedenti in modo tale da limitare superiormente il limite d'errore a cui il modello può arrivare nella fase di classificazione.
# Indichiamo tale valore con <b>C</b>.

# <hr>
# <h4 align="center">SVM Lineare</h4>

# In questa prima applicazione dell'SVM utilizziamo un kernel lineare, ossia cerchiamo quell'iperpiano che massimizza la distanza tra i due cluster di dati. Non siamo in grado, a priori, di determinare quale sia il valore C che massimizza l'accuratezza sul set di testing. Per trovarlo possiamo creare un modello per un insieme di tali valori e scegliere il migliore osservando i risultati ottenuti. <br>
# I valori di C con cui sarà testata l'accuratezza sono <b>{0.001, 0.01, 0.1, 1, 10, 1000, 10000}</b>.

# In[295]:


my_multiclassifier.SVM_linear_analysis((0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))


# <hr>
# <h4 align="center">SVM con kernel RBF</h4>

# Quando i dati non sono linearmente separabili (come in questo caso) è più utile trovare un iperpiano che separi i dati in uno spazio di dimensione superiore. Facendo ciò si aumenta la complessità del calcolo del margine ma si riesce ad aumentare anche lo score del modello.<br>
# Uno dei kernel maggiormente utilizzati nell'ambito del SVM è il cosiddetto <b>Radial basis function kernel (RBF Kernel)</b>. La formulazione standard del kernel è $$ K(x,y) = exp(-\frac{\|x - y\|^2}{2\sigma^2}) $$ che possiamo riscrivere come $$ K(x,y) = exp(-\gamma\|x - y\|^2), \quad \gamma = \frac{1}{2\sigma^2} $$

# Questo tipo di kernel, come si può vedere dalla formula sopra, è formato da due diversi parametri su cui è necessario fare tuning: C e gamma.<br>

# In[296]:


my_multiclassifier.SVM_rbf_analysis((0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000),                                    (0.002, 0.02, 0.2, 2, 20, 200, 2000, 20000))


# A fronte di un costo computazionale maggiore si ottiene un'accuratezza decisamente maggiore rispetto a quella ottenuta con un kernel lineare o utilizzando l'algoritmo KNN.

# <hr>
# <h3 align="center">Regressione logistica</h3>

# Nonostante il nome la regressione logistica, anche chiamata <b>regressione logit</b>, è un algoritmo di classificazione.<br>
# A differenza di altri algoritmi esso fornisce un risultato sulla base di un valore di probabilità e questo implica che il modello utilizzato deve necessariamente restituire un valore sempre compreso tra 0 e 1.<br>
# A tal proposito la regressione logistica utilizza una funzione chiamata <b>Sigmoid function</b>, la cui particolarità è quella di avere codominio compreso tra 0 ed 1.
# $$ h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^- \theta^Tx }  $$ 
# <img src="img/Logistic_curve.svg" width="600">

# <hr>
# <h4 align="center">Maximum Likelihood Estimation (MLE) e Gradient Ascent</h4>

# Il modello che vogliamo costruire è un iperpiano che separa i dati in due gruppi e che denotiamo con la lettera $\theta$. La likelihood (probabilità) di un dati modello $\theta$ per un certo dataset $\vec{X}, \vec{y}$ è definito come $L(\theta) = P(\vec{y}|\vec{X}; \theta)$.<br>
# La MLE ci dice che tra tutti i possibili iperpiani $\theta$ che possiamo trovare, noi cerchiamo quello che <b>massimizza</b> la probabilità per questi dati. Per fare ciò dobbiamo espandere la definizione di likelihood riportata sopra come $$L(\theta) = P(\vec{y}|\vec{X}; \theta) = \prod_i{P(y^{(i)}|x^{i}; \theta)}  $$
# dato che la probabilità di un insieme di dati è uguale al prodotto delle probabilità dei singoli elementi.<br>
# Quando y, cioè l'etichetta, è uguale a 1 allora $P(y^{(i)}|x^{i}; \theta) = h_ \theta (x)$.<br>
# Viceversa quando y = 0 allora abbiamo che $P(y^{(i)}|x^{i}; \theta) = 1 - h_ \theta (x)$

# Inserendo quest'ultima equazione in quella della likelihood e passando in notazione logaritmica otteniamo 
# $$ log L(\theta) = \sum_i{y^{(i)} \cdot log h_\theta(x^{(i)})+(1-y^{i}) \cdot log(1-h_\theta(x^{(i)}))}$$

# L'obiettivo è trovare il valore di $\theta$ (l'iperpiano) per il quale la log-likelihood è massima rispetto a tutti gli altri iperpiani. Essendo tale equazione la somma di un numero arbitrario di parametri non è possibile determinarne il massimo semplicemente calcolandone le derivate parziali. A tal scopo si utilizza un algoritmo di ottimizzazione chiamato <b>Gradient Ascent</b>, che permette di trovare il massimo della funzione per spostandosi sulla superficie da essa definita in maneria graduale e per passaggi successivi.

# <hr>
# <h4 align="center">Analisi dei dati usando la regressione logistica</h4>

# Dopo aver fornito una, seppur superficiale, spiegazione teorica per quanto riguarda la regressione logistica siamo adesso in grado di applicarla al nostro dataset.

# In[190]:


from sklearn.linear_model import LogisticRegression

# Applicazione della regressione logistica
def performLogRegression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    
    # Costruiamo il modello con i dati di training
    clf.fit(X_train, y_train)

    # Testiamo il modello con i dati di testing e stampiamo l'accuratezza
    accuracy = clf.score(X_test, y_test)
    
    return round(accuracy,3)*100
accuracy = performLogRegression(X_train, X_test, y_train, y_test)
print("*** Accuratezza della regressione logistica: {}%".format(accuracy))


# In[ ]:




