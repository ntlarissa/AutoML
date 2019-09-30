## Ce module permet de faire un prétraitement des données afin de supprimer les features inutiles
from visualisation import loadData, Quit
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from collections import Counter
#from assistance import GetPath

#cette fonction effectue le prétraitement
def InitPrepro():
    print("**** Bienvenue dans l\'outil de prétraitement des données ****\n")
    print("     Le prétraitement va consister à:  \n")
    print("        --supprimer les features non pertinents, les NAN et INF, ce qui va permettre de reduire le bruit")
    print("        --transformer les features catégorielles en numériques en utilisant la technique du One Hot Enconding, plusieurs classifiers fonctionnent avec des données numériques")
    print("        --normaliser ou mettre les données au même grandeur d'échelle")

    pathFile="./workspace/datax/dataused.csv"
    ok,data=loadData(pathFile)
    if(ok):
        print("\nCe jeu de données a les features suivantes:\n")
        print(data.columns)
        
        choix=input("\nVoulez vous supprimer certaines features???(y/n) :")
        q=choix.lower()
        if q=='y':
            listfeatures=""
            listfeatures=input("\nBien vouloir indiquer les features à supprimer séparés par la  virgule :")
            #print(listfeatures.split(','))

            data=data.drop(listfeatures.split(','), axis=1) #suppresion de quelques variables
            data=data.replace([np.Inf, np.NINF], np.nan).dropna() # suppresion des nan et Inf
            if(listfeatures!=""):
                print("\nLes colonnes ", listfeatures, "ont été supprimées.")
        target=input("\nQuelle feature correspond à votre critère de sélection???:")
        data=encodage(data,target) # encodage des données
        data=data.replace([np.Inf, np.NINF], np.nan).dropna() # suppresion des nan et Inf
        print("\nLes données manquantes ont été supprimées et les données normalisées.")
        print("\nCe nouveau jeu de données contient ",data.shape[0]," échantillons. Chaque échantillon étant répresentée par ",data.shape[1], " features.\n")
        print(data.head())
        print("\n La statistique des classes:")
        print(Counter(data[target]))
        data.to_csv("./workspace/datax/datapre.csv",index=False)
        choix=input("\nVoulez vous sauvergarder les modifications effectuées???(y/n) :")
        q=choix.lower()
        if q=='y':#sauvergarde des modifications 
            pathFile=input("Entrez le chemin du fichier : ")
            try: 
                data.to_csv(pathFile)
                print("Le fichier a été sauvergardé.")
            except:
                print("Une erreur s'est produite, veuillez réessayer.")
    else:
        print("Une erreur est survenue lors du prétraitement. Bien vouloir réessayer!!!")
    return Quit(1)

def encodage(data,target):
    print(data.info())
    X_train=data.drop(target, axis=1)
    y_train=data[target]
    X1 = X_train.select_dtypes(include=['object'])
    X12 = X_train.select_dtypes(include=['object'])#contient les features encodées avec le label encoding
    for c in list(X1):
        if(X1[c].nunique()>200):
            X1=X1.drop([c],axis=1)
            X12[c]=LabelEncoder().fit_transform(X12[c])
        else:
            X12=X12.drop([c],axis=1)
    ohe = OneHotEncoder()
    #X1_ohe = pd.DataFrame(ohe.fit_transform(X1).toarray())
    cat_columns = X1.columns
    X1_ohe = pd.get_dummies(X1, prefix_sep="__", columns=cat_columns)
    cat_dummies = [col for col in X1_ohe
               if "__" in col
               and col.split("__")[0] in cat_columns]
    X2 = X_train.select_dtypes(exclude=['object'])
    sc = StandardScaler()
    X2_sc = pd.DataFrame(sc.fit_transform(X2),columns=X2.columns)
    X12_le = pd.DataFrame(X12,columns=X12.columns)
    if(X12_le.shape[1]>0):
        X12_le = pd.DataFrame(StandardScaler().fit_transform(X12_le))
    X_train_sc1 = pd.concat([X1_ohe,X12_le,X2_sc], axis=1, sort=False)

    return pd.concat([X_train_sc1,y_train],axis=1,sort=False)


def Quit(data):
    choix=input("Voulez vous continuer avec une autre opération???(y/n) :")
    return choix,data