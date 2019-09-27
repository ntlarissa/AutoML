from visualisation import*
from preprocessing import InitPrepro
from featureengineering import*
from learningmodel import*
from evaluationmetric import*
import argparse
from sklearn.model_selection import train_test_split
import os
import datetime
import shutil

def Choix(i,step):
    if(step==0):
        if i ==1 :
            return InitPrepro()
        elif i == 2:
            return InitFeaEng()
        elif i == 3:
            return InitLeaMod()
        elif i == 4:
            return Quit(0)
        else:
            return "Veuillez choisir la bonne opération!!!"
    elif(step==1):
        if i ==1 :
            return InitFeaEng()
        elif i == 2:
            return InitLeaMod()
        elif i == 3:
            return Quit(data)
        else:
            return "Veuillez choisir la bonne opération!!!"
    elif(step==2):
        if i ==1 :
            return InitLeaMod()
        elif i == 2:
            return Quit(data)
        else:
            return "Veuillez choisir la bonne opération!!!"
    elif(step==3):
        if i ==1 :
            return InitPrepro()
        elif i == 2:
            return Quit(data)
        else:
            return "Veuillez choisir la bonne opération!!!"
def IsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def IsTaux(tx):
    try:
        if((float(tx)>0) and (float(tx)<=100)):
            return True
        else:
            return False
    except ValueError:
        return False
def Process(step):
    global data
    OperationList(step)
    nbre = input("\nEntrez le nombre correspondant à votre opération :")
    print("\n")
    if(IsInt(nbre)):
        ch, niv=Choix(int(nbre),step)
        #print(data.shape)
        q=ch.lower()
        if q=='y':
            Process(niv)
        elif q=='n':
            print("\nMerci d\'avoir utilisé notre outil!!!")
            deleteFile()
        else:
            print("\nVeuillez entrer la bonne lettre!!!")
            Process(0)
    else:
        print("Veuillez entrer un chiffre!!!")
        Process(0)

def OperationList(step):
    if(step==0):
        print('\n')
        print('        Par quelles opérations voulez vous débuter???\n               1 - Prétraitements des données\n               2 - Réduction de dimensions des features ')
        print('               3 - Apprentissage et évaluation du modèle\n               4 - Quitter l\'outil')
    elif(step==1):
        print('\n')
        print('        Par quelles opérations voulez vous continuer???\n               1 - Réduction de dimensions des features ')
        print('               2 - Apprentissage et évaluation du modèle\n               3 - Quitter l\'outil')
    elif(step==2):
        print('\n')
        print('        Par quelles opérations voulez vous continuer???\n    ')
        print('               1 - Apprentissage et évaluation du modèle\n               2 - Quitter l\'outil')
    elif(step==3):
        print('\n')
        print('        Par quelles opérations voulez vous continuer???\n    ')
        print('               1 - Prétraitements des données\n               2 - Quitter l\'outil')
def GetPath():
    if(pathFile==""):
        return False,pathFile
    return True,pathFile

def loadSplitData(chemin,tx):
    try: 
        data = pd.read_csv(chemin,low_memory=False, skipinitialspace=True, quotechar='"')
    except:
        return False,"Le fichier ", chemin, "est introuvable."
    
    if(tx==100):
        datatoused=data
    else:
        data1, datatoused = train_test_split(data, test_size=tx/100)
    #sauvergarde les données dans le workspace courant
    try: 
        #date = datetime.datetime.now()
        nameDossier="./workspace/datax"#+str(date.month)+str(date.day)
        os.makedirs("./workspace", exist_ok=True)
        os.makedirs(nameDossier, exist_ok=True)
        data.to_csv("./workspace/datax/datawhole.csv",index=False)
        datatoused.to_csv("./workspace/datax/dataused.csv",index=False)
        #print(data.head())
    except:
        print("Erreur lors du sauvergarde.")
    return True,datatoused

def deleteFile():
    try:
        shutil.rmtree('./workspace/datax')
    except:
        print("Erreur lors de la suppression!!!!")

#debut
print('**** Bienvenue dans l\'assistance  d\' apprentissage ****')
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="chemin du fichier contenant le dataset")
args = parser.parse_args()

#récupération de l'argument
global pathFile
pathFile=""
data=""
if args.file:
    pathFile=args.file

#Lecture du chemin du dataset
if(pathFile==""):
    pathFile=input("Entrez le chemin du fichier (vous pouvez utiliser wget pour télécharger le fichier): ")

taux = input("\nEntrez le taux(1 à 100) d'utilisation du jeu de données( par exemple 100 si vous voulez utiliser toutes les données)  :")
if(IsTaux(taux)):

    ok,data=loadSplitData(pathFile,float(taux))
    if(ok):
        choix = input("\nVoulez vous visualiser le jeu de données chargé???(y/n)  :")
        q=choix.lower()
        if q=='y':
            InitVisual(data)
        Process(0)
    else:
        print(data)
else:
    print("Veuillez entrer un nombre compris entre 0 et 100!!!")