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

def InitVisual(data):
    print("**** Bienvenue dans l\'outil de visualisation des données ****\n")
    getInfo(data)
    data.describe()
    showMatrix(data)
   

def Quit(data):
    choix=input("Voulez vous continuer avec une autre opération???(y/n) :")
    return choix,data

def loadData(chemin):
    try: 
        data = pd.read_csv(chemin,low_memory=False, skipinitialspace=True, quotechar='"')
        return True,data
    except:
        return False,"le fichier", chemin, "est introuvable."

def showMatrix(data):
    corr=data.corr()
    print ("\nLa figure presente la matrice de corrélation qui permet d'annalyser la dépendance entre les features. ")
    plt.figure(figsize=(25,25))
    sns.heatmap(corr, annot=True,fmt=".1f")
    plt.tight_layout()
    plt.show()

def getInfo(data):
    print("Ce jeu de données contient ",data.shape[0]," échantillons. Chaque échantillon étant répresentée par ",data.shape[1], " features.\n" )
    print("un aperçu des quatre premiers échantillons:" )
    print(data.head())
    print("\n Les features ont les types suivants:")
    print(data.info())

def GetPath(pathFile):
    if(pathFile==""):
        return False,pathFile
    return True,pathFile