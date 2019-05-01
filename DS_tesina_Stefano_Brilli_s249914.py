#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Analisi della qualità del vino Vinho Verde</h1>
# <div align="center"><b>Stefano Brilli - Aprile 2019</b></div>
# <hr><hr>

# <!--bibtex
# 
# @article{database_cit,
#     title = {Modeling wine preferences by data mining from physicochemical properties},
#     author = {P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis},
#     journal = {Decision Support Systems, Elsevier},
#     volume = {47},
#     issue = {4},
#     pages = {547 - 553},
#     year = {2009} 
# }
# -->

# <img src="img/wine.jpg" width="600">

# <h1>Tabella dei contenuti<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduzione" data-toc-modified-id="Introduzione-1">Introduzione</a></span><ul class="toc-item"><li><span><a href="#Strumenti-utilizzati-per-l'analisi" data-toc-modified-id="Strumenti-utilizzati-per-l'analisi-1.1">Strumenti utilizzati per l'analisi</a></span></li></ul></li><li><span><a href="#Esplorazione-e-visualizzazione-dei-dati" data-toc-modified-id="Esplorazione-e-visualizzazione-dei-dati-2">Esplorazione e visualizzazione dei dati</a></span><ul class="toc-item"><li><span><a href="#Distribuzione-dei-valori-delle-feature" data-toc-modified-id="Distribuzione-dei-valori-delle-feature-2.1">Distribuzione dei valori delle feature</a></span></li><li><span><a href="#Correlazione-tra-le-feature" data-toc-modified-id="Correlazione-tra-le-feature-2.2">Correlazione tra le feature</a></span></li></ul></li><li><span><a href="#Analisi-delle-componenti-principali-(PCA)" data-toc-modified-id="Analisi-delle-componenti-principali-(PCA)-3">Analisi delle componenti principali (PCA)</a></span></li><li><span><a href="#Overfitting-e-valutazione-delle-prestazioni-del-modello" data-toc-modified-id="Overfitting-e-valutazione-delle-prestazioni-del-modello-4">Overfitting e valutazione delle prestazioni del modello</a></span></li><li><span><a href="#Validazione-del-modello" data-toc-modified-id="Validazione-del-modello-5">Validazione del modello</a></span></li><li><span><a href="#Classificazione" data-toc-modified-id="Classificazione-6">Classificazione</a></span><ul class="toc-item"><li><span><a href="#Classificatore-Bayesiano" data-toc-modified-id="Classificatore-Bayesiano-6.1">Classificatore Bayesiano</a></span></li><li><span><a href="#K-Nearest-Neighbors-(KNN)" data-toc-modified-id="K-Nearest-Neighbors-(KNN)-6.2">K-Nearest Neighbors (KNN)</a></span></li><li><span><a href="#Support-Vector-Machine-(SVM)" data-toc-modified-id="Support-Vector-Machine-(SVM)-6.3">Support Vector Machine (SVM)</a></span><ul class="toc-item"><li><span><a href="#Il-problema-di-ottimizzazione" data-toc-modified-id="Il-problema-di-ottimizzazione-6.3.1">Il problema di ottimizzazione</a></span></li><li><span><a href="#SVM-Lineare" data-toc-modified-id="SVM-Lineare-6.3.2">SVM Lineare</a></span></li><li><span><a href="#SVM-con-kernel-RBF" data-toc-modified-id="SVM-con-kernel-RBF-6.3.3">SVM con kernel RBF</a></span></li></ul></li><li><span><a href="#Regressione-logistica" data-toc-modified-id="Regressione-logistica-6.4">Regressione logistica</a></span><ul class="toc-item"><li><span><a href="#Maximum-Likelihood-Estimation-(MLE)-e-Gradient-Ascent" data-toc-modified-id="Maximum-Likelihood-Estimation-(MLE)-e-Gradient-Ascent-6.4.1">Maximum Likelihood Estimation (MLE) e Gradient Ascent</a></span></li><li><span><a href="#Analisi-dei-dati-usando-la-regressione-logistica" data-toc-modified-id="Analisi-dei-dati-usando-la-regressione-logistica-6.4.2">Analisi dei dati usando la regressione logistica</a></span></li></ul></li><li><span><a href="#Albero-di-decisione" data-toc-modified-id="Albero-di-decisione-6.5">Albero di decisione</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-6.6">Random Forest</a></span></li><li><span><a href="#Confronto-tra-i-modelli" data-toc-modified-id="Confronto-tra-i-modelli-6.7">Confronto tra i modelli</a></span></li></ul></li><li><span><a href="#Conclusioni" data-toc-modified-id="Conclusioni-7">Conclusioni</a></span></li><li><span><a href="#Bibliografia" data-toc-modified-id="Bibliografia-8">Bibliografia</a></span></li></ul></div>

# <h2 align="center">Introduzione</h2>

# In questo report saranno analizzati dati riguardanti il vino portoghese <b>Vinho Verde</b> nella sua variante rossa con l'obiettivo di predirne la qualità. Il dataset utilizzato è reperibile all'indirizzo http://archive.ics.uci.edu/ml/datasets/Wine+Quality <a id="ref-1" href="#cite-database_cit">[1]</a>.<br>
# Al fine di costruire un classificatore binario il dataset è stato modificato per rendere binaria la variabile di output: le osservazioni aventi un output minore od uguale a 6 sono state etichettate con il <b>valore 0 (bassa qualità)</b> mentre quelle aventi un valore qualitativo di 7 o superiore sono state etichettate con il <b>valore 1 (alta qualità)</b>.<br>
# È chiaro come una mappatura di questo tipo ponga sia i vini con punteggio 6 sia quelli con punteggio 10 allo stesso livello nella categoria di quelli di alta qualità, ma un modello come questo potrebbe permettere ai produttori di avere un iniziale responso sulla qualità del proprio vino, scartando in prima battuta quelli classificati come di bassa qualità ed andando successivamente ad analizzare nel dettaglio i restanti al fine di determinare una scala delle qualità.<br>
# I valori di input sono stati calcolati per mezzo di test fisico-chimici, mentre la qualità finale è stata determinata tramite analisi sensoristiche.<br>
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
# Gli strumenti e le librerie software utilizzate per redigere questa analisi sono:
# <ul>
#     <li><b>Python 3.7.2</b></li>
#     <li><b>Jupyter Notebook</b>, applicazione WEB per la creazione di documenti contenenti codice, testo e grafici</li>
#     <li><b>pandas</b> [<a href="https://pandas.pydata.org/">https://pandas.pydata.org/</a>], libreria software per la manipolazione e l'analisi dei dati</li>
#     <li><b>numpy</b> [<a href="http://www.numpy.org/">http://www.numpy.org/</a>], estensione open source del linguaggio di programmazione Python, che aggiunge supporto per vettori e matrici multidimensionali e di grandi dimensioni e con funzioni matematiche di alto livello con cui operare</li>
#     <li><b>plotly</b> [<a href="https://plot.ly/feed/">https://plot.ly/feed/</a>], insieme di strumenti per l'analisi e la visualizzazione dei dati</li>
#     <li><b>sklearn</b> [<a href="https://scikit-learn.org/stable/">https://scikit-learn.org/stable/</a>], libreria open source di apprendimento automatico per il linguaggio di programmazione Python</li>
# </ul>    

# <hr><hr>
# <h2 align="center">Esplorazione e visualizzazione dei dati</h2>

# Il dataset originale contiene <b>1599 righe</b>, <b>12 colonne</b> (11 di input + 1 di output) e nessuna osservazione parziale o con dati mancanti.

# In[1]:


# Tesina del corso Data Spaces
# Anno accademico 2018/2019
# Stefano Brilli, matricola s249914

import numpy as np
import pandas as pd


from scipy import interp


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

from sklearn import svm
from warnings import simplefilter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (f1_score, recall_score, precision_score, roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression

import math

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

import warnings

from IPython.core.display import display, HTML 

display(HTML("<style>.container { width:100% !important; }</style>"))

warnings.filterwarnings('ignore')

simplefilter(action='ignore', category=FutureWarning)

np.set_printoptions(threshold=sys.maxsize)
# Questa funzione legge i dati dal file elimina le righe che contengono valori non numerici e mappa l'output sui valori 0/1
# Se quality <= 6 --> 0
# Altrimenti quality --> 1
def read_data():
#     path = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/Data Spaces/tesina/wine_quality/'
    df = pd.read_csv('./winequality-red.csv', sep=';')  
    df = df[df.applymap(np.isreal).any(1)]
    df['quality'] = df['quality'].map(lambda x: 1 if x >= 7 else 0) 
    return df


# La tabella seguente mostra le prime dieci osservazioni del dataset, e permette di farsi un'idea su come i dati siano strutturati.

# In[2]:


df = read_data()
df.head(10)


# La tabella seguente è una descrizione dei valori delle feature, che permette di capire come essi siano distribuiti.

# In[3]:


df.drop(columns=['quality']).describe()


# In[4]:


colors = ['#d62728', '#2ca02c']
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


# Come si evince dal grafico a barre soprastante, il dataset è molto sbilanciato in favore dei vini di bassa qualità. Questo potrebbe creare qualche problema nella fase di allenamento del modello, che potrebbe non essere in grado di discriminare così bene i vini di alta qualità a causa del limitato numero degli stessi.

# Nel dettaglio, i dati si distribuiscono come segue:
#   
# <ul>  
#     <li><b>Bassa qualità</b> (1382 su 1599) = <b>86.43%</b></li>
#     <li><b>Alta qualità</b> (217 su 1599) = <b>13.57%</b></li>
# </ul>

# <hr>
# <h3 align="center">Distribuzione dei valori delle feature</h3>

# Utilizzando i box plot è possibile osservare in che modo i valori delle feature siano distribuiti. Essi ci permettono di leggerne valore medio, massimo, minimo e rilevare eventuali outlier, ossia dati che non seguono il trend del gruppo a cui appartengono. Se presenti in numero abbastanza elevato, tali valori possono in alcuni casi rendere più complicata l'analisi, perchè il modello apprenderà pattern che si allontanano dal reale andamento dei dati in analisi.
# I valori sono stati normalizzati al fine di apprezzare meglio la visualizzazione, in quanto alcune feature presentavano valori estremamente alti rendendo il grafico schiacciato verso il basso.

# In[6]:


from plotly import tools
# Normalizzazione dei dati (tra 0 e 1)
X = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
df_norm = pd.DataFrame(x_scaled)

def box_plot_by_feature(X):
    colors = ['#d62728', '#2ca02c']
    y_box_negative = X.loc[X["quality"] == 0]
    y_box_positive = X.loc[X["quality"] == 1]    
    data = []
    X = X.drop(columns=['quality'])
    i = 0
    traces_positions = [[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3], [4,1], [4,2]]
    
    fig = tools.make_subplots(rows=4, cols=3, print_grid=False,                               subplot_titles=('Acidità fissa', 'Acidità volatile', 'Acido citrico', 'Zucchero residuo', "Cloruri",                                               "Diossido di solfuro libero",                                               "Diossido di solfuro totale", "Densità", "pH", "Solfati", "Alcol"))
    
    for column in X.columns:
        trace_0 = go.Box(
            name = "Bassa qualità",
            y = y_box_negative[column].tolist(),
            showlegend=False,
            marker = dict(color = colors[0])
        )        
        fig.append_trace(trace_0, traces_positions[i][0], traces_positions[i][1])        
        trace_1 = go.Box(
            name = "Alta qualità",
            y = y_box_positive[column].tolist(),
            showlegend=False,
            marker = dict(color = colors[1])
        )   
        fig.append_trace(trace_1, traces_positions[i][0], traces_positions[i][1])
        i += 1
    
    
    fig['layout'].update(height=1200, width=1200, title='Qualità rispetto alle feature')    
    iplot(fig)

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


# Per ciascuna feature possiamo disegnare un box plot per vedere come esse si distribuiscano a seconda della qualità del vino. Eventuali similitudini tra i box plot di una stessa caratteristica potrebbero essere sintomo di una feature non così discriminante in termini di valore di output, e quindi di feature inutile nel contesto dell'analisi. Si potrebbe quindi decidere di eliminare tali feature.

# In[7]:


# Distribuzione dell'acidità fissa in base alla qualità del vino
box_plot_by_feature(copy.copy(df))


# Alcune feature non sembrano essere una buona discriminate per la qualità del vino. Tra questi troviamo lo <b>zucchero residuo</b>, che a meno di alcuni outlier nei vini di bassa qualità risulta distribuito quasi allo stesso modo per entrambe le etichette. Lo stesso si può dire per quanto riguarda la feature <b>cloruri</b>, anch'essa non particolarmente discriminante della qualità. 
# <br>Per quanto detto saranno quindi rimosse le suddette caratteristiche.

# <hr>
# <h3 align="center">Correlazione tra le feature</h3>

# Il grafico sottostante riporta la correlazione che intercorre tra la feature. Una correlazione positiva indica che quando la prima feature cresce anche la seconda cresce. Una correlazione negativa indica una tendenza inversa all'aumento: quando la prima aumenta la seconda tende a diminuire.
# Infine un valore vicino allo zero indica una mancanza di correlazione tra le feature.

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


# La maggior parte delle feature sono scarsamente correlate tra loro mentre altre sono correlate, sia direttamente che inversamente. Ad esempio, le feature <b>pH</b> e <b>citric acid</b> sono, come ci si potrebbe aspettare, inversamente correlate in quanto al crescere dell'acidità di una sostanza il pH misurato scende. Per tale motivo anche la feature relativa al pH viene eliminata.

# In[9]:


# Rimozione delle feature che presentano valori troppo distribuiti, in accordo con il box plot precedente
df_norm = df_norm.drop(columns=['residual sugar', 'chlorides', 'pH'])


# <hr><hr>
# <h2 align="center">Analisi delle componenti principali (PCA)</h2>

# L'analisi delle componenti principali (PCA, dall'inglese <b>Principal Components Analysis</b>) è una tecnica utilizzata per ridurre la dimensionalità di un dataset mantenendo il più possibile la varianza tra i dati. Riproiettando i dati lungo un certo numero di assi (di numerosità inferiore al numero di assi originali, da cui il concetto di riduzione della dimensionalità) e scartando le direzioni povere di variabilità si ottiene un dataset più "snello", in cui le colonne sono combinazioni lineari di quelle originali, selezionate dall'algoritmo per massimizzare la variabilità dei dati.<br>
# L'approccio per il calcolo delle componenti principali è il seguente:
# <ul>  
#     <li>i dati vengono standardizzati</li>
#     <li>si calcolano gli autovalori e gli autovettori della matrice di covarianza o di quella di correlazione, oppure si calcola la decomposizione a valore singolo</li>
#     <li>si ordinano gli autovalori in ordine decrescente e si calcolano gli autovettori associati ad essi</li>
#     <li>si costruisce la matrice W dei dati riproiettati utilizzando gli autovettori selezionati</li>
#     <li>si trasforma il dataset originale tramite la matrice W per ottenere un sottospazio Y delle feature di dimensione pari al numero di autovettori utilizzati</li>
# </ul>
# 
# Nel contesto di questo report sono state utilizzate le funzioni messe a disposizioni da Sklearn per il calcolo della PCA, le quali utilizzano il metodo della decomposizione a valore singolo (SVD) per trovare le componenti principali.

# In[10]:


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


# Quando si applica la PCA ad un dataset con molte feature scorrelate è molto improbabile che la proiezione sui primi due assi principali assicuri una distrubuzione dei dati in cluster distinti. Se così fosse potremmo utilizzare solamente queste due componenti per creare i nostri modelli e godremmo della possibilità di visualizzare i margini decisionali su un grafico bidimensionale. In questo caso non è possibile e ne è dimostrazione il grafico sottostante, che mostra una sostenuta sovrapposizione tra i dati quando proiettati sui principali assi.

# In[11]:


doPCA(df_norm, 2, True)
my_pca_data, my_pca_obj = doPCA(df_norm, 5, False)
my_total_pca_data, my_total_pca_obj = doPCA(df_norm, 8, False)


# È quindi necessario decidere il numero di componenti che pensiamo siano sufficienti a spiegare in maniera ottimale il dataset. Il grafico sottostante ci mostra quanta variabilità ogni componente è in grado di darci, insieme ad un totale cumulativo.

# In[12]:


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


# In[13]:


print_explained_variance(my_total_pca_obj.explained_variance_ratio_)


# Un interrogativo che potrebbe sorgere spontaneo è relativo alla quantità di varianza che siamo disposti a perdere per ottenere un miglioramento in termini prestazionali, dovuto alla riduzione della dimensionalità del dataset. In letteratura non esiste una specifica soglia di separazione tra un buon valore di varianza ed uno mediocre o pessimo, anche se alcuni testi suggeriscono un valore che sia sopra a ~85%. Seguendo dunque questo schema e volendo essere certi di avere un buon margine di varianza spiegata si è optato per selezionare le prime cinque componenti, che spiegano una varianza che raggiunge il 90%. È certamente un ottimo compromesso.

# <hr><hr>
# <h2 align="center">Overfitting e valutazione delle prestazioni del modello</h2>

# Quando si costruisce un modello di classificazione si deve evitare che esso apprenda con estrema precisione il pattern seguito dai dati di training, altrimenti non sarà in grado di svolgere in maniera adeguata il suo compito quando gli saranno mostrati dati nuovi. La condizione per cui un classificatore abbia appreso troppo precisamente i dati di training è chiamata <b>overfitting</b> ed è nel nostro interesse combatterla.
# Per fare ciò si possono utilizzare diverse tecniche atte ad allenare il modello utilizzando diversi set di training, in modo tale che il modello non si "abitui" ad uno specifico set ma sia in grado di generalizzare il più possibile con lo scopo di classificare meglio i dati di test, quelli che non sono stati usati per l'apprendimento.<br>
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

# Per ciascuno degli algoritmi di classificazione che useremo sarà quantificata la qualità della previsione utilizzando le seguenti metriche: $$ F1_{score} = 2 \frac{\text{precisione} \times \text{sensibilità}}{\text{precisione} + \text{sensibilità}} $$
# 
# $$ \text{precisione} = \frac{T_{P}}{T_{P} + F_{P}} = \frac{Veri\ positivi}{Totale\ positivi\ previsti} $$
# <br>
# $$\text{sensibilità} = \frac{T_{P}}{T_{P} + F_{N}} = \frac{Veri\ positivi}{Totale\ positivi\ reali}$$
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
# Il valore <b>F1</b> calcola la media pesata tra precisione e sensibilità. Quindi esso tiene in considerazione sia i falsi positivi che che i falsi negativi.

# Esiste poi un'ulteriore metrica usata per valutare le prestazioni del modello quando il dataset non è bilanciato, come in questo caso.
# Si chiama <b>accuratezza pesata</b> e nel caso della classificazione binaria corrisponde alla media aritmetica tra la <i>sensitività</i> (tasso di veri positivi) e <i>specificità</i> (tasso di veri negativi).<br>
# Per ciascun classificatore che andremo a costruire richiederemo che i parametri del modello siano scelti in maniera tale da massimizzare questa misura, e mostreremo anche i valori delle misure descritte sopra.

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

# <hr><hr>
# <h2 align="center">Classificazione</h2>

# <i>Classificare</i> significa riconoscere e discriminare oggetti facenti parti di diverse categorie sulla base di informazioni che sono in nostro possesso. Lo scopo è quello di trovare dei pattern nei dati che siano tipici di ciscuna categoria del dataset. A differenza della regressione, che mira a trovare un output di tipo continuo, la classificazione tenta di etichettare ciascun oggetto utilizzando solamente valori discreti (ad esempio dato un cane determinare la sua razza di appartenenza; le razze sono categorie, non avrebbe senso esprimerle per mezzo di valori continui).
# I modelli costruiti per classificare stanno vivendo un'epoca d'oro perchè sempre di più le decisioni importanti vengono prese da macchine addestrate utilizzando informazioni prese dal contesto decisionale in esame.<br>
# Questo capitolo è dedicato interamente all'analisi del dataset utilizzando diversi algorimi di classificazione:<br>
# <ul>    
#     <li><b>Classificatore Bayesiano</b></li>
#     <li><b>K-Nearest Neighbors</b></li>    
#     <li><b>Support Vector Machine con kernel lineare</b></li>
#     <li><b>Support Vector Machine con kernel RBF</b></li>    
#     <li><b>Regressione Logistica</b></li>
#     <li><b>Albero di decisione</b></li>
#     <li><b>Random Forest</b></li>
# </ul>
# Per ciascuno di essi sarà fornita un'adeguata spiegazione teorica al fine di mettere il lettore, seppur il linguaggio matematico possa essere a volte molto insistente, nelle condizioni di capire i fondamenti del funzionamento e le motivazioni.

# In[50]:


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
          "orange": "rgba(255, 127, 14,.8)",
          "blue":  "rgba(31,119,180,.8)",
          "green": "rgba(0,204,0,.8)",
          "purple": "rgba(255,0,127,.8)",
          "black": "rgba(0,0,0,.8)",
          "azure": "rgba(0,255,255,.8)",
          "yellow": "rgba(241,250,66,.8)"
        }
                
        # Splitto il dataset in due subset: training and testing set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state = 19) 
        
        # Creo un array per contenere tutte le informazioni delle analisi e aggiungo la linea tratteggiata        
        self.roc_traces = []        
    
    # Metodo che aggiunge le informazioni di analisi all'array definito sopra
    def addScores(self, method_name, roc_test_trace, auc_score, color_str):
        
        # Cambio alcuni parametri della traccia
        roc_test_trace['marker']['color'] = self.colors_dict[color_str]
        roc_test_trace['name'] = method_name + " (AUC = " + str(round(auc_score,3)) + ")"        
        
        # Appendo all'array
        self.roc_traces.append(roc_test_trace) 
    
    # Metodo per disegnare la curva ROC comparativa finale
    def plotComparativeROC(self):
        # Aggiungo la traccia della linea tratteggiata
        trace_dash_line = go.Scatter(x=[0, 1], y=[0, 1], 
            mode='lines', 
            line=dict(color='navy', width=2, dash='dash'),
            showlegend=False)
        self.roc_traces.append(trace_dash_line)
        
        # Plotto le curve ROC
        self.plot_roc_curves(self.roc_traces, "Curva ROC comparativa di tutti i modelli")
                
        
    # Metodo per costruire le tracce della curva ROC
    def build_roc_trace(self, fpr, tpr, color, trace_name):                        
        # Singola traccia di curva PR
        trace = go.Scatter(x = fpr, y = tpr, mode = 'lines', name = trace_name, marker = dict(
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
    def plot_roc_curves(self, roc_traces_list, title_str):
        layout_roc = dict(
                title = title_str,
                xaxis = dict(title = "FPR"),
                yaxis = dict(title = "TPR"),
                showlegend=True)
        fig_roc = go.Figure(data=roc_traces_list, layout=layout_roc)
        iplot(fig_roc)    
    
    # Metodo per rappresentare la matrice di confusione
    def plotConfusionMatrix(self, scores, supporto_0, supporto_1):        
        tn, fp, fn, tp = scores
        tn = round(tn/supporto_0, 3)
        fp = round(fp/supporto_0, 3)
        fn = round(fn/supporto_1, 3)
        tp = round(tp/supporto_1, 3)
        
        recall = tp/(tp+fn)
        specificity = 1 - fp/(fp+tn)
        balanced_accuracy = 0.5 * (recall + specificity)
        print("Specificità su testing set: %0.3f" % specificity)
        print("Balanced accuracy su testing set: %0.3f" % balanced_accuracy)

        trace = go.Heatmap(z=[[tn, fp], [fn, tp]],
                           x=["Bassa qualità", "Alta qualità"],
                           y=["Bassa qualità", "Alta qualità"]) 
        layout = dict(
              title="Matrice di confusione",
              yaxis = dict(title="Classe reale"),
              xaxis = dict(title="Classe predetta",)
             )
        figure = go.Figure(data=[trace], layout=layout)
        iplot(figure)
    
    # Metodo per disegnare i barplot di f1, precisione e recall
    def printScores(self, training_report, test_report):
        f1_train = training_report["macro avg"]['f1-score']
        precisione_train = training_report["macro avg"]["precision"]
        recall_train = training_report["macro avg"]["recall"]
        
        f1_test = test_report["macro avg"]['f1-score']
        precisione_test = test_report["macro avg"]["precision"]
        recall_test = test_report["macro avg"]["recall"]
        
        scores = [['F1', f1_train, f1_test],                   ['Precisione', precisione_train, precisione_test],                   ['Sensibilità', recall_train, recall_test]] 
  
        # Dataframe Pandas 
        df = pd.DataFrame(scores, columns = ['Score', 'Training set', 'Testing set'])                
        display(df)
    
    def doAnalysis(self, parameters, clf, clf_id):
        # Costruiamo la griglia per fare tuning dei parametri
        # Vogliamo il miglior modello in termini di balanced accuracy
        grid = GridSearchCV(clf, parameters, scoring='balanced_accuracy')
        
        # Nomi delle etichette
        target_names = ['Bassa qualità', 'Alta qualità']
        
        # Facciamo tuning dei parametri passando il dataset di training
        grid.fit(self.X_train, self.y_train)
        
        # Parametri del miglior modello trovato
        print("Parametri del miglior classificatore trovato: {}".format(grid.best_params_))               
        
        # Costruiamo un classificatore utilizzando i migliori parametri trovati
        
        # Classificatore Bayesiano
        if clf_id == 0:            
            best_clf = GaussianNB(var_smoothing = grid.best_params_['var_smoothing'])        
        
        # KNN
        if clf_id == 1:            
            best_clf = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'], weights = grid.best_params_['weights'],                                        p = grid.best_params_['p'])
        # SVM Lineare
        elif clf_id == 2:
            best_clf = svm.SVC(C = grid.best_params_['C'], kernel='linear', gamma='auto', probability=True,                               class_weight = grid.best_params_['class_weight'])
        
        # SVM con kernel rbf
        elif clf_id == 3:
            best_clf = svm.SVC(kernel='rbf', C = grid.best_params_['C'], gamma = grid.best_params_['gamma'], probability=True,                               class_weight = grid.best_params_['class_weight'])
            
        
        # Regressione Logistica
        elif clf_id == 4:
            best_clf = LogisticRegression(penalty = grid.best_params_['penalty'], class_weight = grid.best_params_['class_weight'],                                          tol = grid.best_params_['tol'])
        
        # Albero di decisione
        elif clf_id == 5:
            best_clf = DecisionTreeClassifier(criterion = grid.best_params_['criterion'],                                              class_weight = grid.best_params_['class_weight'],                                              splitter = grid.best_params_['splitter'],                                              presort = grid.best_params_['presort'])
            
        # Random Forest
        elif clf_id == 6:            
            best_clf = RandomForestClassifier(n_estimators = grid.best_params_['n_estimators'],                                              criterion = grid.best_params_['criterion'],                                              class_weight = grid.best_params_['class_weight'],                                              bootstrap = grid.best_params_['bootstrap'])                        
                        

        # Fitto i dati di training utilizzando il miglior classificatore
        best_clf.fit(self.X_train, self.y_train)
        
        # Calcolo ed estraggo i risultati ottenuti con il miglior classificatore trovato ed utilizzato sul dataset di training
        # Stampo solo gli score inerenti la classe 1
        y_train_predict = best_clf.predict(self.X_train)
        training_report = classification_report(self.y_train, y_train_predict, output_dict=True, target_names = target_names)                
        
        # Calcolo le previsione sul set di testing          
        y_test_predict = best_clf.predict(self.X_test)                
        
        # Stampo i risultati ottenuti        
        test_report = classification_report(self.y_test, y_test_predict, output_dict=True, target_names = target_names)         
        supporto_0 = test_report["Bassa qualità"]["support"]
        supporto_1 = test_report["Alta qualità"]["support"]         
        
        # Plotto gli andamenti di f1, precisione e recall
        self.printScores(training_report, test_report)
        
        # Matrice di confusione
        #print(confusion_matrix(self.y_test, y_test_predict))
        print("Numero di elementi di classe 0 nel testing set: {}".format(supporto_0))
        print("Numero di elementi di classe 1 nel testing set: {}".format(supporto_1))
        self.plotConfusionMatrix(confusion_matrix(self.y_test, y_test_predict).ravel(), supporto_0, supporto_1)
        
        # Calcolo gli score per la curva ROC
        y_score_train = best_clf.predict_proba(self.X_train)
        y_score_test = best_clf.predict_proba(self.X_test)
        
        # Calcolo valori tpr, fpr per curva ROC
        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_score_train[:,1])
        fpr_test, tpr_test, _ = roc_curve(self.y_test, y_score_test[:, 1])
        
        # Calcolo area sottesa dalla curva ROC per training e testing set
        auc_train = roc_auc_score(self.y_train, y_score_train[:, 1])
        auc_test = roc_auc_score(self.y_test, y_score_test[:, 1]) 
        
        # Costruisco le due tracce per il grafico della curva ROC
        trace_train, trace_dash_line = self.build_roc_trace(fpr_train, tpr_train, "blue", "Training set")
        trace_test, trace_dash_line = self.build_roc_trace(fpr_test, tpr_test, "orange", "Testing set")
        
        # Plotto le curve ROC
        self.plot_roc_curves([trace_train, trace_test, trace_dash_line], "Curva ROC di training e testing set")
        
        
        # Stampo il valore dell'area sotto la curva
        auc_string_train = "AUC del training set: %0.3f" % auc_train
        auc_string_test = "AUC del testing set: %0.3f" % auc_test        
        print(auc_string_train)
        print(auc_string_test)
        
        return trace_test, auc_test, best_clf
    
    # Metodo per l'analisi tramite Naive Bayes classifier
    def NB_analysis(self):
        clf = GaussianNB()
       
        # Parametri del classificatore
        parameters = {
            'var_smoothing': (1e-7, 1e-8, 1e-9, 3e-9, 7e-9)
        }
        
        trace, auc, _ = self.doAnalysis(parameters, clf, 0)                
        self.addScores("Bayes Classifier Gaussiano", trace, auc, "yellow") 

    # Metodo per l'analisi del dataset usando l'algoritmo KNN
    def KNN_analysis(self, param_range): 
        clf = KNeighborsClassifier() 
        
        # Parametri del classificatore
        parameters = {
            # Valori di K passati come parametro
            'n_neighbors': param_range, 
            
            # Tipologia di classificatore KNN
            'weights' : ('uniform', 'distance'),
            
            # Il valore di p indica la metrica utilizzata per valutare la distanza
            # p = 1 L1, p = 2 L2
            'p': (1, 2, 3)
        }        
        trace, auc, _ = self.doAnalysis(parameters, clf, 1)                
        self.addScores("KNN", trace, auc, "blue")                        
        
    
    # Metodo per la l'analisi del dataset usando l'algoritmo SVM con kernel lineare
    def SVM_linear_analysis(self, param_range):                
        clf = svm.SVC(kernel='linear', gamma='auto', probability=True)
        
        # Parametri del classificatore
        parameters = {
            # Valori di C passati come parametro
            'C': param_range,
            'class_weight': ('balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:7}, {0:1, 1:10},
                            {0:1, 1:20}, {0:1, 1:50}, {0:1, 1:100})
        }        
        trace, auc, _ = self.doAnalysis(parameters, clf, 2)
        self.addScores("SVM con kernel lineare", trace, auc, "orange")
        
    # Metodo utile alla costruzione del modello tramite utilizzo del kernel rbf 
    def SVM_rbf_analysis(self, c_range, gamma_range):
        clf = svm.SVC(kernel='rbf', gamma='auto', probability=True)
        
        # Parametri del classificatore
        parameters = {
            # Valori di C passati come parametro
            'C': c_range,
            'gamma': gamma_range,
            'class_weight': ('balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:7}, {0:1, 1:10},
                            {0:1, 1:20}, {0:1, 1:50}, {0:1, 1:100})
        }        
        trace, auc, _ = self.doAnalysis(parameters, clf, 3)
        self.addScores("SVM con kernel RBF", trace, auc, "green")
        
    # Applicazione della regressione logistica
    def performLogRegression(self):
        clf = LogisticRegression()  
        
        # Parametri del classificatore
        parameters = {            
            'penalty': ('l1', 'l2'),
            'class_weight': ('balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:7}, {0:1, 1:10},
                            {0:1, 1:20}, {0:1, 1:50}, {0:1, 1:100}),
            'tol': (1e-4, 5e-5, 2e-4, 1e-5, 1e-6, 2e-6, 3e-4)
        }        
        trace, auc, _ = self.doAnalysis(parameters, clf, 4)
        self.addScores("Regressione Logistica", trace, auc, "purple")
        
    # Metodo per costruire un classificatore basato sull'algoritmo random forest
    def treeClassifier(self):
        clf = DecisionTreeClassifier()
        
        parameters = {
            'criterion': ('gini', 'entropy'),
            'splitter': ('best', 'random'),
            'presort': (True, False),
            'class_weight': ('balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:7}, {0:1, 1:10},
                            {0:1, 1:20}, {0:1, 1:50}, {0:1, 1:100})
            
        }        
        trace, auc, best_clf = self.doAnalysis(parameters, clf, 5)
        self.addScores("Albero di decisione", trace, auc, "azure")
        
        feature_importances = best_clf.feature_importances_
        colonne = copy.copy(df_norm).drop(columns=['quality']).columns
        d = {}
        for feature, importance in zip(colonne, feature_importances):
            d[feature] = [importance]
        data_frame = pd.DataFrame.from_dict(data=d, columns=['Importanza della feature'], orient='index')
        display(data_frame)
    
    # Metodo per costruire un classificatore basato sull'algoritmo random forest
    def performRandomForest(self):
        clf = RandomForestClassifier()
        
        parameters = {
            'n_estimators': (10, 20, 25, 30, 40, 100),
            'criterion': ('gini', 'entropy'),
            'class_weight': ('balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:7}, {0:1, 1:10},
                            {0:1, 1:20}, {0:1, 1:50}, {0:1, 1:100}),
            'bootstrap': (True, False)
        }        
        trace, auc, best_clf = self.doAnalysis(parameters, clf, 6)
        self.addScores("Random Forest", trace, auc, "black")  
        
        feature_importances = best_clf.feature_importances_
        colonne = copy.copy(df_norm).drop(columns=['quality']).columns
        d = {}
        for feature, importance in zip(colonne, feature_importances):
            d[feature] = [importance]
        data_frame = pd.DataFrame.from_dict(data=d, columns=['Importanza della feature'], orient='index')
        display(data_frame)


# In[51]:


my_multiclassifier = Multiclassifier(df, df_norm)


# <hr>
# <h3 align="center">Classificatore Bayesiano</h3>

# I cosiddetti <i>metodi Naive Bayes</i>, letteralmente <i>metodi ingenui di Bayes</i> sono un insieme di algoritmi di tipo <b>supervisionato</b> che operano per mezzo del teorema di Bayes con l'assunzione di indipendenza condizionale tra ciascuna coppia di osservazioni (ossia, il valore di un'osservazione non è influenzato dal valore dell'altra). Il <b>teorema di Bayes</b> si può riassumere nella seguente equazione:
# 
# $$ \begin{align}\begin{aligned}P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)\\\Downarrow\\\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),\end{aligned}\end{align} $$
# 
# dove $ y $ è l'output e $ x_1, ..., x_n $ sono le osservazioni. Questo algoritmo risulta semplice da implementare e comodo da utilizzare, perchè richiede pochi dati di training per funzionare in maniera discreta e non consuma troppe risorse computazionali.<br>
# I classificatori basati su questo algoritmo si differenziano in base alla probabilità (stimata a priori) che l'analista assegna ai dati di input.<br>
# Assumendo a priori che i dati siano distribuiti secondo una certa distribuzione, che è compito dell'analista individuare, si possono costruire modelli diversi.
# In questo caso assumiamo che i dati osservati siano distribuiti secondo una normale, con media $ \mu = 0 $ e varianza $ \sigma^2  = 1$ (i dati erano stati normalizzati in precedenza). In altre parole il termine dentro la produttoria sarà $$ P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right) $$

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>var_smoothing</i>: porzione della più grande varianza tra tutte le feature che è aggiunta per stabilità di calcolo</li>
# </ul>

# In[21]:


my_multiclassifier.NB_analysis()


# Il classificatore risulta efficace sia nella predizione di vini di bassa qualità che di vini di alta qualità, con i primi marchiati correttamente più di <b>9 volte su 10</b>. Nell'ottica di minimizzare il numero di falsi positivi potremmo ritenerci molto soddisfatti del modello, il quale permetterebbe ad un'azienda produttrice una perdita minima in termini di prodotto scartato (meno del ~1% è predetto positivo erroneamente).<br>
# Il problema potrebbe sorgere nella situazione inversa, infatti più di <b>3 volte su 10</b> un vino di alta qualità è stato scartato dal modello. Questo potrebbe creare problemi alle aziende con un capitale non così elevato e che vivono di quantità, mentre potrebbe essere accettabile per aziende grandi che puntano di più sulla qualità.<br>
# In generale <b>il modello è soddisfacente</b>.

# <hr>
# <h3 align="center">K-Nearest Neighbors (KNN)</h3>

# Il KNN è un algoritmo utilizzato per individuare e classificare elementi sulla base di pattern spaziali. Tra gli algoritmi utilizzati nel Machine Learning è quello più semplice da implementare e da comprendere ed il funzionamento è il seguente: i voti dei k vicini del campione determinano il gruppo a cui assegnare l'oggetto in esame. Per k = 1 il campione viene assegnato al gruppo dell'oggetto a lui più vicino.
# Per scegliere il valore di k più adeguato è necessario valutare la tipologia e la quantità di dati che si possiede. Infatti scegliere un valore di k elevato garantisce una migliore resistenza al rumore ma rende meno risconoscibili i margini di separazione. 

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>n_neighbors</i>: numero di vicini da tenere in considerazione</li>
#     <li><i>p</i>: potenza per la metrica di Minkowski (per p = 1 questo corrisponde alla distanza L1 mentre per p = 2 corrisponde alla metrica L2, cioè la distanza euclidea)</li>
#     <li><i>weights</i>: funzione di peso (uniforme, distanza)</li>
# </ul>

# In[22]:


# Uso il KNN testando tutti i valori di K compresi tra 1 e 40 con passo 1
my_multiclassifier.KNN_analysis(range(1, 40, 1))


# Gli score ottenuti ci mostrano come il modello sia andato in overfitting durante la fase di training, mostrando punteggi globali più bassi durante la fase di testing.<br>
# La matrice di confusione mostra un incremento nel numero di previsioni vere negative ma anche un incremento nel numero di previsioni false positive. Così come detto per il classificatore precedente, uno score di questo tipo potrebbe giovare ad aziende che hanno come obiettivo la qualità più che la quantità, ma potrebbe allo stesso tempo danneggiare le aziende che puntano sulla qualità.<br>
# In definitiva <b>questo modello potrebbe non essere soddisfacente</b> in determinati contesti applicativi.

# <hr>
# <h3 align="center">Support Vector Machine (SVM)</h3>

# L'SVM è un algoritmo supervisionato che si pone come obiettivo quello di trovare l'iperpiano che massimizzi la distanza tra le osservazioni di classi diverse. Il calcolo formulato in prima battuta non è banale, perchè presuppone la presenza del vettore $ \omega $ dei pesi e dunque di prodotti scalari. Come spiegato poco sotto è possibile ricondurre il problema originale ad un problema derivato eliminando la dipendenza dal vettore dei pesi.
# 
# 
# <img src="img/svm_1.png" width="400">

# <hr>
# <h4 align="center">Il problema di ottimizzazione</h4>

# Se i dati sono perfettamente separabili nello spazio allora è dimostrato come sia possibile trovare l'iperpiano di separazione in un numero finito di passi. Ciò è garantito dal <b>teorema di convergenza del Perceptron</b>.<br>
# A questo punto viene naturale chiedersi quale dei possibili iperpiani dobbiamo scegliere per separare i dati secondo le etichette. L'SVM per definizione ricerca l'iperpiano che massimizza la distanza tra i due gruppi di elementi (si veda l'immagine sopra).<br>
# $$ \text{ Formulazione primale: } \min\frac{1}{2}||\omega||^2 \text{ subject to: } y^{(t)}(\omega \cdot x^{(t)} + b) \geq k, \ t=1,...,n $$ <br>
# Si tratta di un <b>problema di ottimizzazione quadratico e con vincolo</b>. Esso può essere risolto utilizzando il metodo dei <b>moltiplicatori di Lagrange</b>, ed essendo un problema quadratico si tratta di trovare il minimo globale (unico) di una superficie paraboloidale.
# Calcolando le derivate parziali della formulazione primale otteniamo la formulazione alternativa del problema. Essa è definita come $$ \text{ Formulazione alternativa: } \max\text{-}\frac{1}{2}\sum_{ij}{\alpha_i\alpha_j y_i y_j \vec{x_i} \cdot \vec{x_j}} + \sum_i{\alpha_i} \text{ subject to: } \sum_i{\alpha_i y_i} = 0 \text{, } \alpha_i > 0 $$

# Formulando il problema in questo modo siamo in grado di calcolare la soluzione elimando la dipendenza dal vettore omega dei pesi e dal bias b. 
# L'unica dipendenza rimasta per la risoluzione è quella relativa ai coefficienti alfa.
# Essi sono per lo più uguali a zero, e gli unici diversi da zero sono quelli relativi agli elementi presenti sui margini, che prendono il nome di <b>support vectors</b>, da cui il nome dell'algoritmo.

# Nel caso in cui i dati non siano separabili linearmente, come in questo caso, possiamo pensare di risolvere il problema di ottimizzazione permettendo ad alcuni elementi del dataset di essere dalla parte sbagliata del margine, o addirittura dalla parte sbagliata dell'iperpiano. Questo si formalizza aggiungengo un ulteriore vincolo alle formulazioni precedenti in modo tale da limitare superiormente il limite d'errore a cui il modello può arrivare nella fase di classificazione.
# Indichiamo tale valore con <b>C</b>.

# <hr>
# <h4 align="center">SVM Lineare</h4>

# In questa prima applicazione dell'SVM utilizziamo un kernel lineare, ossia cerchiamo quell'iperpiano che massimizza la distanza tra i due cluster di dati. Non siamo in grado, a priori, di determinare quale sia il valore C che massimizza l'accuratezza sul set di testing. Per trovarlo possiamo creare un modello per un insieme di tali valori e scegliere il migliore osservando i risultati ottenuti. <br>

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>C</i>: parametro C dell'SVM</li>
#     <li><i>class_weight</i>: peso dato a ciascuna delle due classi, sulla base del numero di campioni che ne fanno parte</li>
# </ul>

# In[23]:


my_multiclassifier.SVM_linear_analysis((0.07,0.08,0.09,0.1,0.2,0.22,0.25,0.3,0.4))


# Ciò che salta immediatamente all'occhio guardando la matrice di confusione è l'elevato valore di <b>veri positivi</b> che sono stati predetti nella fase di testing (<b>~89% delle previsioni</b>). Bisogna valutare se sia desiderabile o meno un valore di <b>falsi positivi che si attesti al 22%</b>. Ciò significa che per ogni 5 osservazioni di un vino di bassa qualità, una viene predetta come di alta qualità. Potrebbe in effetti essere un buon compromesso considerando il già indicato valori di veri positivi.<br>
# Considerando anche che il numero di <b>falsi negativi</b> registrato è <b>piuttosto basso (11.5%)</b> e che quello di <b>veri negativi</b> è <b>sufficientemente alto (78.4%)</b> possiamo dire che il modello trovato riesce a mediare piuttosto bene tutte le possibili situazioni. Ne è dimostrazione la curva ROC, la cui area sottesa raggiunge il 91.6%.<br>
# Per quanto detto <b> il modello è soddisfacente</b>.

# <hr>
# <h4 align="center">SVM con kernel RBF</h4>

# Quando i dati non sono linearmente separabili (come in questo caso) è più utile trovare un iperpiano che separi i dati in uno spazio di dimensione superiore. Facendo ciò si aumenta la complessità del calcolo del margine ma si riesce ad aumentare anche lo score del modello.<br>
# Uno dei kernel maggiormente utilizzati nell'ambito del SVM è il cosiddetto <b>Radial basis function kernel (RBF Kernel)</b>. La formulazione standard del kernel è $$ K(x,y) = exp(-\frac{\|x - y\|^2}{2\sigma^2}) $$ che possiamo riscrivere come $$ K(x,y) = exp(-\gamma\|x - y\|^2), \quad \gamma = \frac{1}{2\sigma^2} $$

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>C</i>: parametro C dell'SVM</li>
#     <li><i>Gamma</i>: parametro gamma dell'SVM</li>
#     <li><i>class_weight</i>: peso dato a ciascuna delle due classi, sulla base del numero di campioni che ne fanno parte</li>
# </ul>

# In[24]:


my_multiclassifier.SVM_rbf_analysis((0.07,0.08,0.09,0.1,0.2,0.3,0.4),                                    (0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.9,1.0))


# Il classificatore si comporta molto bene quando ha di fronte un vino di alta qualità, il quale viene classificato correttamente nella quasi totalità dei casi. In un contesto in cui ciò che conta è il vino di alta qualità un modello come questo sarebbe eccellente perchè riesce a rilevare quasi sempre l'alta qualità quando presente. Bisogna però valutare attentamente il caso contrario. Questo classificatore sbaglia quasi <b>3 volte su 10</b> la classificazione quando si trova a dover decidere l'etichetta da assegnare ad un vino di bassa qualità.
# Osservando però la curva ROC possiamo concludere l'elevata bontà del modello nel prendere le sue decisioni al variare della soglia decisionale.<br>
# In generale <b>il modello è soddisfacente</b>.

# <hr>
# <h3 align="center">Regressione logistica</h3>

# Nonostante il nome possa far pensare ad un algoritmo di regressione, la regressione logistica è un metodo utilizzato per la classificazione e deve il nome ad una connessione storica con la regressione lineare.<br>
# L'equazione seguente è quella utilizzata per valutare i valori di input (i concetti teorici sono spiegati poco sotto) e prende il nome di <b>Sigmoid function</b>.
# $$ h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^- \theta^Tx },\ \theta= [b, \omega_1, \omega_2, \cdots \omega_d]$$ 
# <img src="img/Logistic_curve.svg" width="600">

# <hr>
# <h4 align="center">Maximum Likelihood Estimation (MLE) e Gradient Ascent</h4>

# Il modello che vogliamo costruire è un iperpiano che separa i dati in due gruppi e che denotiamo con la lettera $\theta$. La likelihood (probabilità) di un dato modello $\theta$ per un certo dataset $\vec{X}, \vec{y}$ è definito come $L(\theta) = P(\vec{y}|\vec{X}; \theta)$.<br>
# La MLE ci dice che tra tutti i possibili iperpiani $\theta$ che possiamo trovare, noi cerchiamo quello che <b>massimizza</b> la probabilità per questi dati. Per fare ciò dobbiamo espandere la definizione di likelihood riportata sopra come $$L(\theta) = P(\vec{y}|\vec{X}; \theta) = \prod_i{P(y_i|x_i; \theta)}  $$
# dato che la probabilità di un insieme di dati è uguale al prodotto delle probabilità dei singoli elementi.<br>
# Quando y, cioè l'etichetta, è uguale a 1 allora $P(y_i|x_i; \theta) = h_ \theta (x)$.<br>
# Viceversa quando y = 0 allora abbiamo che $P(y_i|x_i; \theta) = 1 - h_ \theta (x)$<br>
# <br>Si ottiene quindi $$ L(\theta) = \prod_i{(1 - h(X_i;\theta))^{y_i}h(X_i;\theta)^{(1-y_i)}} $$
# 

# Passando in notazione logaritmica otteniamo 
# $$ log L(\theta) = \sum_i{y_i  log(1 - h(X_i;\theta)) + (1 - y_i)log(h(X_i;\theta))}$$

# L'obiettivo è trovare il valore di $\theta$ (l'iperpiano) per il quale la log-likelihood è massima rispetto a tutti gli altri iperpiani. Per fare ciò siamo soliti calcolare la derivata della quantità e poi porla uguale a zero. In questo caso non è possibile procedere così perchè l'equazione è la somma di un numero arbitrario di termini e non ha una forma chiusa. Per trovare il massimo dell'equazione si utilizza la tecnica chiamata <b>gradient ascent</b> (che prende il nome di <i>gradient descent</i> quando si vuole risolvere un problema di minimizzazione). <br>
# Essa consiste nell'aggiornare il valore di $\theta$ ad ogni iterazione, aggiungendo ad essa la derivata della funzione calcolata nel punto $\theta$. In questo modo si cerca di raggiungere il punto massimo della funzione per step successivi, senza passare dalla risoluzione analitica, che come abbiamo visto non è possibile fare.<br>
# Questo aggiornamento si può scrivere matematicamente come
# $$ \theta^{(t)} \rightarrow \theta^{(t-1)} + \eta f'(\theta) |_{\theta\ =\ \theta^{(t-1)}} $$
# dove $\eta$ è un termine di regolazione chiamato <b>learning rate</b> e serve a quantificare la dimensione dello spostamento lungo la curva.<br><br>
# Applicando il concetto di gradient ascent alla regressione logistica otteniamo
# $$ \theta^{(t)} \rightarrow \theta^{(t-1)} + \eta \nabla log L(\theta) |_{\theta\ =\ \theta^{(t-1)}} $$
# <br>
# Una volta calcolati il vettore dei pesi $ \omega $ ed il bias, per classificare una nuova osservazione non dobbiamo fare altro che calcolare l'output della funzione sigmoide quando l'input è il valore delle feature dell'osservazione moltiplicata scalarmente con il vettore dei pesi. Un output sopra la soglia dello <b>0.5</b> classificherà l'osservazione con l'etichetta <b>1</b>, mentre uno sotto tale soglia sarà classificato con etichetta <b>0</b>. Il valore 0.5 si riferisce, come detto, ad una probabilità maggiore o minore il 50% che l'osservazione faccia parte della classe positiva.
# 
# L'ultimo concetto teorico che è bene citare riguarda la <b>regolarizzazione</b>. Utilizzando la procedura appena trattata c'è il rischio che il vettore $\omega$ dei pesi diventi estremamente grande, e questo porterebbe ad una situazione di overfitting, in quanto la funzione sigmoide restituirebbe valori di output il più delle volte uguali a 0 od 1, e questa non è una situazione desiderabile. Per porre rimedio al problema è possibile <i>penalizzare</i> il modello ogni volta che sbaglia in modo da, appunto, regolarizzare il processo di apprendimento. Durante l'analisi con regressione logistica useremo sia la penalizzazione basata su norma L2 (chiamata <b>ridge</b>) che quella su norma L1 (anche detta <b>lasso</b>).

# <hr>
# <h4 align="center">Analisi dei dati usando la regressione logistica</h4>

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>penalty</i>: funzione di perdita usata e che si vuole minimizzare</li>
#     <li><i>tol</i>: tolleranza per determinare quando l'algoritmo debba interrompersi</li>
#     <li><i>class_weight</i>: peso dato a ciascuna delle due classi, sulla base del numero di campioni che ne fanno parte</li>
# </ul>

# In[25]:


my_multiclassifier.performLogRegression()


# I risultati ottenuti sono all'incirca gli stessi rilevati utilizzando il classificatore precedente.<br>
# Per tale motivo concludiamo che <b>il modello è soddisfacente</b>.

# <hr>
# <h3 align="center">Albero di decisione</h3>

# Gli alberi di decisione sono algoritmi supervisionati e non parametrici usati sia per la regressione che per la classificazione. L'obiettivo è quello di creare un modello che sia in grado di assegnare le etichette ai dati sulla base di una regola di decisione dedotta dalle caratteristiche dei dati di training. Tra i vantaggi nell'utilizzo degli alberi di decisione troviamo:
# <ul>
#     <li>Sono molto semplici da visualizzare e da spiegare, anche a persone che non sono del mestiere</li>
#     <li>Con gli alberi è possibile trattare dati categorici senza la necessità di creare variabili dummy.</li>
#     <li>Richiedono una poco esosa preparazione dei dati</li>
# </ul>
# Essi portano con sè anche alcuni svantaggi:
# <ul>
#     <li>In genere gli alberi di decisione non forniscono un'accuratezza elevata come altri algoritmi classificativi.</li>
#     <li>Gli alberi tendono ad essere poco robusti: un piccolo cambiamento nei dati può comportare un enorme variazione dell'albero.</li>
#     <li>I modelli basati su alberi di decisione tendono a fornire una non-buona generalizzazione dei dati, ossia vanno in overfitting.</li>
#     <li>In presenza di dataset troppo sbilanciati verso una determinata classe il modello potrebbe risultare molto affetto da bias.</li>
# </ul
# 
# In questo paragrafo sarà costruito un modello basato su un albero di decisione semplice mentre nel prossimo capitolo costruiremo un modello che utilizza metodi di aggregazione.

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>criterion</i>: funzione per misurare la qualità dello split di un nodo (gini, entropy)</li>
#     <li><i>presort</i>: parametro per fare un ordinamento prima di applicare l'algoritmo</li>
#     <li><i>splitter</i>: strategia usata per effettuare lo split di un nodo (random, best)</li>
#     <li><i>class_weight</i>: peso dato a ciascuna delle due classi, sulla base del numero di campioni che ne fanno parte</li>
# </ul>

# In[52]:


my_multiclassifier.treeClassifier()


# La tabella qui sopra riporta l'importanza di ciascuna delle feature nel processo di costruzione degli alberi che costituiscono il modello dell'albero di decisione. La feature <b>alcool</b> è chiaramente quella dominante che ha determinato la costruzione di questo modello, mentre tutte le altre hanno assunto un'importanza da due a tre volte minore. <br>

# In scikit-learn l'importanza di un nodo $ j $ è calcolata come $$ ni_j = w_j C_j - w_{left(j)}C_{left(j)}- w_{right(j)}C_{right(j)} $$<br>
# dove $ w_j $ è il numero di sample al nodo $ j $, $ C_j $ è l'impurità a questo nodo, mentre $ left(j) $ e $ right(j) $ sono i rispettivi nodi figlio.

# L'importanza della $i$-esima feature è quindi calcolata come $$ fi_i = \frac{\sum_{j : \text{node j splits on feature i}} ni_j}{\sum_{j \in \text{all nodes}} ni_j} $$

# I risultati ottenuti sono simili a quelli visti con il modello basato su KNN. Il numero di falsi negativi è piuttosto alto a fronte di un elevato valore di veri negativi. In generale però il modello non pare essere in grado di svolgere un buon lavoro al variare delle soglie decisionali, come evidenziato dalla curva ROC.<br>
# Concludiamo quindi che <b>questo modello potrebbe non essere soddisfacente</b> in determinati contesti applicativi.

# <hr>
# <h3 align="center">Random Forest</h3>

# Gli alberi di decisione appena trattati soffrono di <b>alta varianza</b>. In altre parole se noi dividessimo il training set in due parti in maniera casuale e costruissimo due modelli ad albero, noteremmo delle differenze in termini di accuratezza raggiunta. Di contro saremmo in grado di ottenere all'incirca gli stessi risultati su subset diversi abbassando la varianza della procedura. 

# Il termine <b>bagging</b> significa <b>boostrap aggregation</b> ed è una procedura utilizzata allo scopo di ridurre la varianza del metodo utilizzato per l'analisi. Si basa sull'assunto che dato un insieme di variabili indipendenti ciascuna di variaza $ \sigma^2 $, la varianza media dell'insieme è data da $ \sigma^2/n $. Questo significa che applicare un operatore di media su un dataset ne abbassa la varianza. Di conseguenza un metodo naturale per ridurre la varianza è quello di prendere diversi set da una data popolazione, costruire un modello predittivo usando ciascun training set e calcolare la media delle accuratezze ottenute.

# Fatte queste premesse possiamo definire il random forest come un metodo di classificazione basato sull'utilizzo di una moltitudine di alberi di decisione e la cui previsione si basa su quella che è stata la classe predetta il più delle volte dai singoli alberi da cui è composto.
# La particolarità di questo classificatore risiede nella maniera con cui i nodi vengono splittati: invece di scegliere la miglior feature e dividere di conseguenza il nodo sulla base di essa viene scelta la migliore feature all'interno di un ristretto sottoinsieme di feature. Questo porta alla creazione di una moltitudine di alberi tutti diversi tra loro, perchè costruiti aggiungendo una componente random alla fase di creazione. 
# È possibile aggiungere un'ulteriore componente randomica all'albero, scegliendo in maniera casuale le soglie decisionali ai nodi piuttosto che trovare le migliori possibili.

# I parametri su cui faremo tuning per questo algoritmo sono:
# <ul>
#     <li><i>criterion</i>: funzione per misurare la qualità dello split di un nodo (gini, entropy)</li>
#     <li><i>bootstrap</i>: utilizzo della tecnica del bootstrap oppure no</li>
#     <li><i>n_estimators</i>: numero di alberi nella foresta</li>
#     <li><i>class_weight</i>: peso dato a ciascuna delle due classi, sulla base del numero di campioni che ne fanno parte</li>
# </ul>

# In[49]:


my_multiclassifier.performRandomForest()


# La caratteristica più importante è stata quella che descrive il tasso di <b>alcool</b>, mentre quella meno importante è stata quella associata al <b>diossido di solfuro libero</b>. Notiamo come la feature alcool sia la più importante sia in questo caso che in quello dell'albero di decisione, anche se in quest'ultimo essa aveva assunto una più marcata importanza. In questo caso invece l'importanza sembra essersi distribuita lungo tutte le feature, come lecito aspettarsi da un modello basato sulla selezione random delle feature.

# L'importanza delle feature viene calcolata alla stessa maniera di come fatto nel caso dell'albero di decisione, con la differenza che qui le importanze vengono mediate rispetto al numero totale di alberi costruiti.

# La tonalità rosso scuro del settore delle previsioni vere negative ci dice chiaramente come questo modello rasenti la perfezione nel classificare i vini di bassa qualità, il che lo rende il modello quasi-ideale in un contesto in cui i falsi negativi contano. Non dobbiamo farci ingannare dall'elevato valore di <i>falsi positivi</i> della matrice di confusione: essa indica gli score relativi ad una specifica soglia decisionale. Dobbiamo invece considerare la curva ROC del modello, che mostra valori molto elevati sia sul training set che sul testing set.<br>
# Detto questo concludiamo che <b>il modello è estremamente soddisfacente</b>, attestandosi come il migliore trovato in questa analisi.

# <hr>
# <h3 align="center">Confronto tra i modelli</h3>

# Allo scopo di confrontare il comportamento generale dei sette algoritmi di classificazione utilizzati in questa analisi, sono sotto riportate le curve ROC calcolate sul testing set.

# In[28]:


my_multiclassifier.plotComparativeROC()


# In base alle considerazioni fatte per ciascuno dei modelli costruiti, e sulla base delle curve ROC mostrate sopra, possiamo decretare il classificatore basato su random forest il migliore trovato, il quale ha registrato score molto elevati. Tra i peggiori troviamo, all'incirca a pari merito, il KNN e l'albero di decisione semplice.

# <hr><hr>
# <h2 align="center">Conclusioni</h2>

# Per mezzo di questa analisi siamo riusciti a costruire un insieme di modelli in grado di prevedere la qualità del Vino <b>Vinho Verde</b> soltanto sulla base di otto sue caratteristiche ed ottenendo, almeno per alcuni dei modelli, risultati soddisfacenti.<br>
# Alcuni di essi ci hanno permesso di rilevare con estrema precisione i vini di bassa qualità, altri ci hanno garantito ottime prestazioni nell'evitare le false previsioni. Abbiamo capito quindi che non esiste il modello perfetto ma che è compito dell'analista trovare il giusto compromesso in base al dominio d'analisi in cui si trova. Gli strumenti a nostra disposizione sono stati fondamentali per la buona riuscita dell'analisi, perchè ci hanno permesso di concentrarci sui risultati piuttosto che sull'implementazione. Il linguaggio Python è risultato estremamente appropriato per via della sua semplicità d'uso ed immediatezza applicativa. Infine l'ambiente Jupyter ha semplificato la stesura del report permettendo una armoniosa convivenza di codice ed esposizione tecnica e grafica.

# L'analisi è stata condotta operando diverse scelte sul dataset, quali ad esempio l'eliminazione di colonne ridondanti od inutili perchè poco discriminanti. La PCA ci ha permesso di eliminare alcune dimensioni pur mantenendo una elevata variabilità e il tuning dei parametri ci è servito per trovare la configurazione ottimale dei modelli. Tali impostazioni sono solo alcune delle possibili e di conseguenza siamo stati in grado di utilizzare solamente una piccola parte dei possibili classificatori. Non abbiamo dunque la pretesa di aver trovato i migliori modelli possibili, ma dobbiamo comunque ritenerci soddisfatti dei risultati raggiunti.

# <hr><hr>
# <h2 align="center">Bibliografia</h2>
# 
# <a id="cite-database_cit"/><sup><a href=#ref-1>[^]</a></sup>P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 2009. _Modeling wine preferences by data mining from physicochemical properties_.
# 
# 
