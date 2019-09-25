## Ce module permet de faire un feature engineering afin de reduire par exemple la dimmentsion des données
from visualisation import Quit,loadData
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import os.path
import time

def InitFeaEng():
    print ( "**** Bienvenue dans l\'outil du features engineering ****\n")
    

    pathFile="./workspace/datax/datapre.csv"
    if not os.path.isfile(pathFile):
        print("      Le prétraitement n'a pas été effectué dans cette session")
        pathFile="./workspace/datax/dataused.csv"
    ok,data=loadData(pathFile)
    if(ok):
        if(not IsPreprocessing(data)):
            print("\n       Les données doivent être prétraitées...")
            return Quit(3)
        print("    La réduction des dimensions permet d'améliorer les performances du classifier.")
        print("    Nous pouvons distinguer deux approches pour l'implementer:")
        print("           **La selecton des features qui consiste à choisir un sous ensemble optimal de features.")
        print("                 1) La matrice de corrélation, le temps de traitement est moins d'une minuite")
        print("                 2) gain d'information, le temps de traitement est  ",0.000277*data.shape[0]*data.shape[1]," minuites")
        print("           **L'extraction  des features qui consiste à créer un sous ensemble optimal de features.")
        print("                 3) L'analyse en composantes principales, le temps de traitement est moins d'une minuite ")
        print("                 4) L'analyse discriminate linéaire, le temps de traitement est moins d'une minuite ")
        
        number=input("Quelles techniques souhaiteriez vous utiliser??? Entrez le numéro (1 à 4) correspondant à la technique ou 5 si aucune technique ne sera utilisée ")
        
        if(IsInt(number)):
            #print(data.shape)
            new_data=Technique(int(number),data)
            #print(new_data)
            
            return Quit(1)
        else:
            print("Veuillez entrer un chiffre!!!")
            return Quit(1)
    else:
        print("Une erreur est survenue lors du traitement. Bien vouloir réessayer!!!")
    return Quit(1)

def Quit(data):
    choix=input("Voulez vous continuer avec une autre opération???(y/n) :")
    return choix,data

def Technique(i,data):
    if i ==1 :
        return MaCorr(data)
    elif i == 2:
        return GaIn(data)
    elif i == 3:
        return AnCoPr(data)
    elif i == 4:
        return AnDiLi(data)
    else:
        return "Veuillez choisir la bonne technique!!!"

def MaCorr(data):
    new_data=""
    print("Ce traitement peut mettre en moyenne ",0.0135*data.shape[0]," secondes.")
    new_data=""
    len=data.shape[1]
    taille=len/4
    X_train=data.iloc[:,0:len-1]
    Y_train=data.iloc[:,len-1]
    Y_train_ohe=pd.DataFrame(OneHotEncoder().fit_transform(Y_train.values.reshape(-1, 1)).toarray())
    data_sc = pd.concat([X_train,Y_train_ohe], axis=1, sort=False)

    data_corr=data_sc.corr()
    #on supprime les données sur la lignes
    for c in Y_train_ohe:
        data_corr=data_corr.drop([c], axis=0)

    resultat=np.zeros((len-1,), dtype='f')
    #on calclule la moyenne ces coefficients
    for c in Y_train_ohe:
        resultat+=data_corr[c]/Y_train_ohe.shape[1]
    resultat=pd.DataFrame(resultat,index=X_train.columns)
    resultat=resultat.sort_values(by = 0, ascending = False)
    print("\n Le classement des features en fonction des coefficients de corrélation:")
    print(resultat)
    nbre_ss_ens=input("Entrez le nombre de features à selectionner: ")
    if(IsInt(nbre_ss_ens)):
        taille=int(nbre_ss_ens)
    new_data=pd.concat([X_train[resultat.index[0:taille]],Y_train],axis=1,sort=False) 
    new_data.to_csv("./workspace/datax/datafeaGI.csv",index=False)
    print(new_data.head())
    data.to_csv("./workspace/datax/datafeaMC.csv")
    return new_data

def GaIn(data):
    new_data=""
    len=data.shape[1]
    taille=len/4
    X_train=data.iloc[:,0:len-1]
    Y_train=data.iloc[:,len-1]
    Y_train_ohe=pd.DataFrame(OneHotEncoder().fit_transform(Y_train.values.reshape(-1, 1)).toarray())
    GI=np.zeros((len-1,), dtype='f')
    #on calclule la moyenne ces coefficients
    for c in Y_train_ohe:
        GI += mutual_info_classif(X_train, Y_train_ohe[c])/Y_train_ohe.shape[1]
    
    resultat=pd.DataFrame(GI,index=X_train.columns)
    resultat=resultat.sort_values(by = 0, ascending = False)
    print("\n Le classement des features en fonction du gain d'information:")
    print(resultat)
    nbre_ss_ens=input("Entrez le nombre de features à selectionner: ")
    if(IsInt(nbre_ss_ens)):
        taille=int(nbre_ss_ens)
    new_data=pd.concat([X_train[resultat.index[0:taille]],Y_train],axis=1,sort=False) 
    new_data.to_csv("./workspace/datax/datafeaGI.csv",index=False)
    print(new_data.head())
    return new_data

def AnCoPr(data):
    new_data=""
    taille=2
    nbre_ss_ens=input("Entrez la taille du sous ensemble: ")
    if(IsInt(nbre_ss_ens)):
        taille=int(nbre_ss_ens)
    len=data.shape[1]
    pca = PCA(n_components=taille)
    X_train=data.iloc[:,0:len-1]
    Y_train=data.iloc[:,len-1]
    print(X_train.shape)
    #pca.fit(X_train)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
    new_data=pd.concat([X_train_pca,Y_train],axis=1,sort=False)
    new_data.to_csv("./workspace/datax/datafeaACP.csv",index=False)
    print(new_data.head())
    return new_data

def AnDiLi(data):
    len=data.shape[1]
    new_data=""
    X_train=data.iloc[:,0:len-1]
    Y_train=data.iloc[:,len-1]
    lda = LDA(shrinkage='auto', solver='eigen') 
    Y_train_en=LabelEncoder().fit_transform(Y_train)
    X_lda = pd.DataFrame(lda.fit_transform(X_train, Y_train_en  ))
    new_data=pd.concat([X_lda,Y_train],axis=1,sort=False)
    new_data.to_csv("./workspace/datax/datafeaADL.csv",index=False)
    print(new_data.head())
    return new_data

def IsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

#verifie si les données ont été encodées
def IsPreprocessing(data):
    lent=data.shape[1]
    X_train=data.iloc[:,0:lent-1]
    X1 = X_train.select_dtypes(include=['object'])
    if(len(list(X1))>0):
        return False
    return True