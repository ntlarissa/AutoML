## Ce module permet de faire un prétraitement des données afin de supprimer les features inutiles
from visualisation import loadData, Quit
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler

def InitPrepro():
    print("**** Bienvenue dans l\'outil de prétraitement des données ****\n")
    #global data
    #print(data)
    ok,data=loadData('./data/UNSW_NB15_training-set_selected.csv')
    if(ok):
        choix=input("Voulez vous continuer avec le jeux de données actuel???(y/n) :")
        q=choix.lower()
        if q=='y':
            choix=input("Voulez vous supprimer certaines variables???(y/n) :")
            q=choix.lower()
            if q=='y':
                number=""
                number=input("Bien vouloir indiquer les numéros des colonnes à supprimer séparés par la  virgule :")
                data=data.drop(['id','label'], axis=1)
                data=data.replace([np.Inf, np.NINF], np.nan).dropna()
                if(number!=""):
                    print("Les colonnes ", number, "ont été supprimées.")
                data=encodage(data)
                print("Les données manquantes ont été supprimées et les données centrées réduites.")
                choix=input("\nVoulez vous sauvergarder les modifications effectuées???(y/n) :")
                q=choix.lower()
                if q=='y':
                    pathFile=input("Entrez le chemin du fichier : ")
                    try: 
                        data.to_csv(pathFile)
                        print("Le fichier a été sauvergardé.")
                        print("Les nouvelles dimensions: ",data.shape)
                        print(data.head())
                    except:
                        print("Une erreur s'est produite, veuillez réessayer.")
            elif q=='n':
                data=data.replace([np.Inf, np.NINF], np.nan).dropna()
                data=encodage(data)
                print("Les données manquantes ont été supprimées et les données centrées réduites.")
            else:
                print("\nVeuillez entrer la bonne lettre!!!")
        elif q=='n':
            pathFile=input("Entrez le chemin du fichier (vous pouvez utiliser wget pour télécharger le fichier): ")
            #KDDTrain+.csv UNSW_NB15_training-set_selected.csv ./data/Wednesday-workingHours.pcap_ISCX.csv
            ok,data=loadData(pathFile)
            if ok:
                data=data.replace([np.Inf, np.NINF], np.nan).dropna()
                print("Les données manquantes ont été supprimées.\n")
                choix=input("Voulez vous supprimer certaines variables???(y/n) :")
                q=choix.lower()
                if q=='y':
                    number=""
                    number=input("Bien vouloir indiquer les numéros des colonnes à supprimer séparés par la  virgule :")
                    data=data.drop(data.columns[number], axis=1)
                    if(number!=""):
                        print("Les colonnes ", number, "ont été supprimées.")
            else:
                print("Une erreur s\'est produite: ",data)
        else:
            print("\nVeuillez entrer la bonne lettre!!!")
    else:
        pathFile=input("Entrez le chemin du fichier (vous pouvez utiliser wget pour télécharger le fichier): ")
            #KDDTrain+.csv UNSW_NB15_training-set_selected.csv ./data/Wednesday-workingHours.pcap_ISCX.csv
        ok,data=loadData(pathFile)
        if ok:
            data=data.replace([np.Inf, np.NINF], np.nan).dropna()
            print("Les données manquantes ont été supprimées.\n")
            choix=input("Voulez vous supprimer certaines variables???(y/n) :")
            q=choix.lower()
            if q=='y':
                number=""
                number=input("Bien vouloir indiquer les numéros des colonnes à supprimer séparés par la  virgule :")
                data=data.drop(data.columns[number], axis=1)
                if(number!=""):
                    print("Les colonnes ", number, "ont été supprimées.")
        else:
            print("Une erreur s\'est produite: ",data)
    return Quit()

def encodage(data):
    X_train=data.drop(['attack_cat'], axis=1)
    y_train=data['attack_cat']
    X1 = X_train.select_dtypes(include=['object'])
    ohe = OneHotEncoder()
    X1_ohe = pd.DataFrame(ohe.fit_transform(X1).toarray())
    X2 = X_train.select_dtypes(exclude=['object'])
    sc = StandardScaler()
    X2_sc = pd.DataFrame(sc.fit_transform(X2))
    X_train_sc1 = pd.concat([X1_ohe,X2_sc], axis=1, sort=False)

    encoder = LabelEncoder()
    Y_train_1 = pd.DataFrame(encoder.fit_transform(y_train))

    return pd.concat([X_train_sc1,Y_train_1],axis=1,sort=False)


