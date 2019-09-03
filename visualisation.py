## Ce module permet de faire une visualisation des données
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import readline
import seaborn as sns


data="null"
etape=1

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

def InitVisual():
    global data
    print("**** Bienvenue dans l\'outil de visualisation des données ****\n")
    #readline.set_completer_delims(' \t\n;')
    #readline.parse_and_bind("tab: complete")
    #readline.set_completer(complete)
    pathFile=input("Entrez le chemin du fichier (vous pouvez utiliser wget pour télécharger le fichier): ")
    #KDDTrain+.csv UNSW_NB15_training-set_selected.csv ./data/Wednesday-workingHours.pcap_ISCX.csv
    ok,data=loadData(pathFile)
    if ok:
        getInfo(data)
        data.describe()
        showMatrix(data)
    else:
        print("Une erreur s\'est produite: ",data)
    return Quit()

def Quit():
    choix=input("Voulez vous continuer avec une autre opération???(y/n) :")
    return choix

def loadData(chemin):
    try: 
        data = pd.read_csv(chemin,low_memory=False)
        return True,data
    except:
        return False,"le fichier", chemin, "est introuvable."

def showMatrix(data):
    corr=data.corr()
    #print (corr)
    plt.figure(figsize=(25,25))
    sns.heatmap(corr, annot=True,fmt=".1f")
    plt.tight_layout()
    plt.show()

def getInfo(data):
    print("Les dimensions: ",data.shape)
    print(data.head())
    print(data.info())
