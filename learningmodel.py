## Dans ce module nous allons entraîner et évaluer le modèle 
from visualisation import Quit,loadData
import os.path
from featureengineering import IsPreprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras

#vérifie si c'est un entier
def IsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

#vérifie si le nombre est compris entre 1 et 100
def IsTaux(tx):
    try:
        if((float(tx)>0) and (float(tx)<=100)):
            return True
        else:
            return False
    except ValueError:
        return False

#La fonction principale
def InitLeaMod():
    print( "**** Bienvenue dans l\'outil d\'apprentissage ****\n")
    print("         Avec quel jeu de données voulez vous continuer:")
    if  os.path.isfile("./workspace/datax/datapre.csv"):
        print("             1 - celui ayant toutes les features??")
    if  os.path.isfile("./workspace/datax/datafeaMC.csv"):
        print("             2 - celui ayant été selectionné avec la matrice de corrélation??")
    if  os.path.isfile("./workspace/datax/datafeaGI.csv"):
        print("             3 - celui ayant été selectionné avec le gain d'information??")
    if  os.path.isfile("./workspace/datax/datafeaACP.csv"):
        print("             4 - celui ayant été extrait  avec l'analyse en composantes principales??")
    if  os.path.isfile("./workspace/datax/datafeaADL.csv"):
        print("             5 - celui ayant été extrait  avec l'analyse discriminate linéaire ??")
    nbre = input("\nEntrez le nombre correspondant à votre choix :")
    if(IsInt(nbre)):
        nbre=int(nbre)
        #scanne des fichiers existants
        if(nbre==1):
            pathFile="./workspace/datax/datapre.csv"
        elif(nbre==2):
            pathFile="./workspace/datax/datafeaMC.csv"
        elif(nbre==3):
            pathFile="./workspace/datax/datafeaGI.csv"
        elif(nbre==4):
            pathFile="./workspace/datax/datafeaACP.csv"
        elif(nbre==5):
            pathFile="./workspace/datax/datafeaADL.csv"
        else:
            print("Veuillez entrer un chiffre compris entre 1 et 5!!!")
            return Quit(2)
        #vérifie l'existance du fichier
        if not os.path.isfile(pathFile):
            print("\n      Le prétraitement ou la réduction de dimensions n'a pas été effectué dans cette session")
            pathFile="./workspace/datax/dataused.csv"
        ok,data=loadData(pathFile)
        if(ok):
            #vérifie si les données  ont été étraitées
            if(not IsPreprocessing(data)):
                print("\n       Les données doivent être prétraitées...")
                return Quit(3)
            print("\n           Les algorithmes suivants ont été intégrés:")
            print("                1 - Arbre de décision")
            print("                2 - Forêt aléatoire")
            print("                3 - K plus proches voisins")
            print("                4 - Machine à vecteurs de support")
            print("                5 - Classification bayésienne")
            print("                6 - Réseaux de neurones")
            
            listalgo=""
            #récupére la liste des choix
            listalgo=input("\n               Sélectionner un ou plusieurs algorithmes séparés par la virgule: ")
            listalgo=listalgo.split(',')
            taux = input("\nEntrez le taux(1 à 100) de division du jeu de données( par exemple 10 si vous voulez utiliser 10% pour les données tests)  :")
            if(IsTaux(taux)):
                cv = input("\nEntrez le nombre d'itération pour la validation croisée: ")
                if(IsInt(cv)):
                    X,Y1=split_data(data)
                    nclaase=Y1.nunique()#récupére le nombre de classe
                    encoder = LabelEncoder()
                    encoder.fit(Y1)
                    xaxis=encoder.classes_#récupére le nom des classes
                    Y = pd.DataFrame(encoder.transform(Y1))#encode les données de sorties

                    #plot
                    fig, ax = plt.subplots(figsize=(10,7))
                    plt1=plt.subplot(121)
                    plt2=plt.subplot(122)
                    index = np.arange(nclaase)
                    bar_width = 0.15
                    opacity = 0.8

                    #apprentissage et évaluation
                    for c in listalgo:
                        if(IsInt(c)):
                            
                            if(int(c)==1):
                                #calcul des mètriques
                                accuracy_score_DT,precision_mean_score_DT,precision_DT,recall_DT=DT(X,Y,int(cv),float(taux),nclaase)
                                #affichage
                                rects3 = plt1.bar(index+ 2*bar_width, precision_DT, bar_width,
                                                    alpha=opacity,
                                                    color='c',
                                                    label='DT')
                                plt1.plot(xaxis, [accuracy_score_DT]*nclaase,color='black', label="ACC-DT")
                                
                                rects31 = plt2.bar(index+ 2*bar_width, recall_DT, bar_width,
                                                    alpha=opacity,
                                                    color='c',
                                                    label='DT')
                                
                            elif(int(c)==2):
                                accuracy_score_RF,precision_mean_score_RF,precision_RF,recall_RF=RF(X,Y,int(cv),float(taux),nclaase)
                                rects1 = plt1.bar(index, precision_RF, bar_width,
                                                        alpha=opacity,
                                                        color='b',
                                                        label='RF')
                                #plt.subplot(122)
                                rects11 = plt2.bar(index, recall_RF, bar_width,
                                                        alpha=opacity,
                                                        color='b',
                                                        label='RF')
                                plt2.plot(xaxis, [accuracy_score_RF]*nclaase,color='black', label="ACC-RF")
                            elif(int(c)==3):
                                accuracy_score_KNN,precision_mean_score_KNN,precision_KNN,recall_KNN=KNN(X,Y,int(cv),float(taux),nclaase)
                                rects4 = plt1.bar(index + 3*bar_width, precision_KNN, bar_width,
                                            alpha=opacity,
                                            color='r',
                                            label='KNN')
                                plt1.plot(xaxis, [accuracy_score_KNN]*nclaase,color='b', label="ACC-KNN")
                                rects41 = plt2.bar(index + 3*bar_width, recall_KNN, bar_width,
                                            alpha=opacity,
                                            color='r',
                                            label='KNN')
                            elif(int(c)==4):
                                accuracy_score_SVM,precision_mean_score_SVM,precision_SVM,recall_SVM=SVM(X,Y,int(cv),float(taux),nclaase)
                                rects2 = plt1.bar(index + bar_width, precision_SVM, bar_width,
                                                alpha=opacity,
                                                color='g',
                                                label='SVM')
                                
                                rects21 = plt2.bar(index + bar_width, recall_SVM, bar_width,
                                                alpha=opacity,
                                                color='g',
                                                label='SVM')
                                plt2.plot(xaxis, [accuracy_score_SVM]*nclaase,color='r', label="ACC-SVM")
                            elif(int(c)==5):
                                accuracy_score_NB,precision_mean_score_BN,precision_NB,recall_NB=GB(X,Y,int(cv),float(taux),nclaase)
                                rects5 = plt1.bar(index+ 4*bar_width, precision_NB, bar_width,
                                                alpha=opacity,
                                                color='y',
                                                label='NB')
                                plt1.plot(xaxis, [accuracy_score_NB]*nclaase,color='g', label="ACC-NB")
                                rects51 = plt2.bar(index+ 4*bar_width, recall_NB, bar_width,
                                                alpha=opacity,
                                                color='y',
                                                label='NB')
                            elif(int(c)==6):
                                RN(X,Y,int(cv),float(taux),nclaase)
                    #affichage des mètriques
                    plt.subplot(121)
                    plt.xlabel('Les différents types d\'attaques')
                    plt.ylabel('Précision')
                    plt.title('Précision par types d\'attaques')
                    plt.xticks(index + bar_width, xaxis,rotation='15')
                    plt.legend()

                    plt.tight_layout()
                    plt.grid()

                    plt.subplot(122)
                    plt.xlabel('Les différents types d\'attaques')
                    plt.ylabel('Recall')
                    plt.title('Recall par types d\'attaques')
                    plt.xticks(index + bar_width, xaxis,rotation='15')
                    plt.legend()

                    plt.tight_layout()
                    plt.grid()
                    plt.show()

                    
                else:
                    print("Veuillez entrer un nombre!!!")
            else:
                print("Veuillez entrer un nombre compris entre 0 et 100!!!")
        else:
            print("Veuillez entrer un chiffre!!!")
            return Quit(2)
    return Quit(0)

#séparer les données de sortie et les données d'entrée
def split_data(data):
    target1=list(data)
    listtarget=""
    for c in target1:
            listtarget+=','+c
    target=listtarget.split(',')[-1]
    #print(listtarget.split(',')[-1])
    X=data.drop([target], axis=1)
    Y=data[target]
    return X,Y

#classifier decision tree
def DT(X,Y,cv,tx,nbreClasse):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tx/100)
    if(cv==1):
        #définir les paramètres
        param_grid = {'max_depth': [10, 50, 100, 500, 1000]}
        #initialiser le classifier
        tree_clf = DecisionTreeClassifier()
        #entrainer et choisir le meilleur classifier
        grid_search = GridSearchCV(tree_clf, param_grid, cv=5
                                , scoring='neg_mean_squared_error')

        grid_search.fit(X_train, Y_train.values.ravel())
        clf4 = grid_search.best_estimator_
        start = time. time()
        #prédire les rsultats
        y_pred_DT = clf4.predict(X_test)
        end = time. time()
        DT_time=end - start
        accuracy_score_DT=accuracy_score(Y_test, y_pred_DT)*100
        precision_score_DT= precision_score(Y_test, y_pred_DT, average=None)*100
        recall_score_DT=recall_score(Y_test, y_pred_DT, average=None)*100
        precision_mean_score_DT=precision_score(Y_test, y_pred_DT, average="micro")*100
        print("DT test accuracy: ", accuracy_score_DT)
        print("DT test précision : ", precision_mean_score_DT)
        print("\nDT test précision par classe: ",precision_score_DT)
        print("\nDT test recall par classe: ", recall_score_DT)
        print("\n DT le temps de traitement: ",DT_time)
        return accuracy_score_DT,precision_mean_score_DT,precision_score_DT,recall_score_DT
    else:
        #mélanger et diviser les données
        sss = StratifiedShuffleSplit(n_splits=cv, random_state=0)
        precision_soe_DT=[0]*nbreClasse
        recall_soe_DT=[0]*nbreClasse
        accuracy_score_DT=0
        precision_mean_score_DT=0
        start = time. time()

        for train_index, test_index in sss.split(X,Y):
            X_train_sc, X_test_sc = X.iloc[train_index], X.iloc[test_index]
            Y_train_en, Y_test_en = Y.iloc[train_index], Y.iloc[test_index]
            param_grid = {'max_depth': [10, 50, 100, 500, 1000]}
            tree_clf = DecisionTreeClassifier()
            grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='neg_mean_squared_error')

            grid_search.fit(X_train_sc, Y_train_en.values.ravel())
            clf4 = grid_search.best_estimator_
            y_pred_DT = clf4.predict(X_test_sc)
            end = time. time()
            accuracy_score_DT+=accuracy_score(Y_test_en, y_pred_DT)*100/cv
            precision_score_DT= precision_score(Y_test_en, y_pred_DT, average=None)
            recall_score_DT=recall_score(Y_test_en, y_pred_DT, average=None)

            precision_mean_score_DT+=precision_score(Y_test_en, y_pred_DT, average="micro")*100/cv
            precision_soe_DT+=precision_score_DT*100/cv
            recall_soe_DT+=recall_score_DT*100/cv
            
        print("DT test accuracy : ",accuracy_score_DT )
        print("DT test précision : ", precision_mean_score_DT)
        print("DT test précision par classe: ",precision_soe_DT )
        print("DT test recall par classe: ",recall_soe_DT )
        DT_time=end - start
        print("DT time en secondes: ",DT_time)
        return accuracy_score_DT,precision_mean_score_DT,precision_soe_DT,recall_soe_DT

def RF(X,Y,cv,tx,nbreClasse):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tx/100)
    if(cv==1):
        param_grid = {'max_depth': [ 800, 1000], 'n_estimators': [ 500,400, 600]}
        randomForest_clf = RandomForestClassifier()
        grid_search = GridSearchCV(randomForest_clf, param_grid, cv=5, scoring='neg_mean_squared_error')


        grid_search.fit(X_train, Y_train.values.ravel())
        clf4 = grid_search.best_estimator_
        start = time. time()
        y_pred_RF = clf4.predict(X_test)
        end = time. time()
        DT_time=end - start
        accuracy_score_RF=accuracy_score(Y_test, y_pred_RF)*100
        precision_score_RF= precision_score(Y_test, y_pred_RF, average=None)*100
        recall_score_RF=recall_score(Y_test, y_pred_RF, average=None)*100
        precision_mean_score_RF=precision_score(Y_test, y_pred_RF, average="micro")*100
        print("RF test accuracy: ", accuracy_score_RF)
        print("RF test précision : ", precision_mean_score_RF)
        print("\RF test précision par classe: ",precision_score_RF)
        print("\nRF test recall par classe: ", recall_score_RF)
        print("\n RF le temps de traitement: ",DT_time)
        return accuracy_score_RF,precision_mean_score_RF,precision_score_RF,recall_score_RF
    else:
        sss = StratifiedShuffleSplit(n_splits=cv, random_state=0)
        precision_soe_RF=[0]*nbreClasse
        recall_soe_RF=[0]*nbreClasse
        accuracy_score_RF=0
        precision_mean_score_RF=0
        start = time. time()

        for train_index, test_index in sss.split(X,Y):
            X_train_sc, X_test_sc = X.iloc[train_index], X.iloc[test_index]
            Y_train_en, Y_test_en = Y.iloc[train_index], Y.iloc[test_index]

            param_grid = {'max_depth': [ 800, 1000], 'n_estimators': [ 500,400, 600]}
            randomForest_clf = RandomForestClassifier()
            grid_search = GridSearchCV(randomForest_clf, param_grid, cv=5, scoring='neg_mean_squared_error')

            grid_search.fit(X_train_sc, Y_train_en.values.ravel())
            clf4 = grid_search.best_estimator_
            y_pred_RF = clf4.predict(X_test_sc)
            end = time. time()
            accuracy_score_RF+=accuracy_score(Y_test_en, y_pred_RF)*100/cv
            precision_score_RF= precision_score(Y_test_en, y_pred_RF, average=None)
            recall_score_RF=recall_score(Y_test_en, y_pred_RF, average=None)

            precision_mean_score_RF+=precision_score(Y_test_en, y_pred_RF, average="micro")*100/cv
            precision_soe_RF+=precision_score_RF*100/cv
            recall_soe_RF+=recall_score_RF*100/cv
            
        print("\n RF test accuracy : ",accuracy_score_RF )
        print("\n RF test précision : ", precision_mean_score_RF)
        print("\n RF test précision par classe: ",precision_soe_RF )
        print("\n RF test recall par classe: ",recall_soe_RF )
        DT_time=end - start
        print("\n RF time en secondes: ",DT_time)
        return accuracy_score_RF,precision_mean_score_RF,precision_soe_RF,recall_soe_RF

def KNN(X,Y,cv,tx,nbreClasse):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tx/100)
    if(cv==1):
        param_grid = {'n_neighbors': [3, 5, 7, 10, 15, 20, 30]}
        knn_clf = KNeighborsClassifier()
        grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring='neg_mean_squared_error')


        grid_search.fit(X_train, Y_train.values.ravel())
        clf4 = grid_search.best_estimator_
        start = time. time()
        y_pred_KNN = clf4.predict(X_test)
        end = time. time()
        KNN_time=end - start
        accuracy_score_KNN=accuracy_score(Y_test, y_pred_KNN)*100
        precision_score_KNN= precision_score(Y_test, y_pred_KNN, average=None)*100
        recall_score_KNN=recall_score(Y_test, y_pred_KNN, average=None)*100
        precision_mean_score_KNN=precision_score(Y_test, y_pred_KNN, average="micro")*100
        print("KNN test accuracy: ", accuracy_score_KNN)
        print("KNN test précision : ", precision_mean_score_KNN)
        print("\nKNN test précision par classe: ",precision_score_KNN)
        print("\nKNN test recall par classe: ", recall_score_KNN)
        print("\n KNN le temps de traitement: ",KNN_time)
        return accuracy_score_KNN,precision_mean_score_KNN,precision_score_KNN,recall_score_KNN
    else:
        sss = StratifiedShuffleSplit(n_splits=cv, random_state=0)
        precision_soe_KNN=[0]*nbreClasse
        recall_soe_KNN=[0]*nbreClasse
        accuracy_score_KNN=0
        precision_mean_score_KNN=0
        start = time. time()

        for train_index, test_index in sss.split(X,Y):
            X_train_sc, X_test_sc = X.iloc[train_index], X.iloc[test_index]
            Y_train_en, Y_test_en = Y.iloc[train_index], Y.iloc[test_index]
            param_grid = {'n_neighbors': [3, 5, 7, 10, 15, 20, 30]}
            knn_clf = KNeighborsClassifier()
            grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring='neg_mean_squared_error')


            grid_search.fit(X_train_sc, Y_train_en.values.ravel())
            clf4 = grid_search.best_estimator_
            y_pred_KNN = clf4.predict(X_test_sc)
            end = time. time()
            accuracy_score_KNN+=accuracy_score(Y_test_en, y_pred_KNN)*100/cv
            precision_score_KNN= precision_score(Y_test_en, y_pred_KNN, average=None)
            recall_score_KNN=recall_score(Y_test_en, y_pred_KNN, average=None)

            precision_mean_score_KNN+=precision_score(Y_test_en, y_pred_KNN, average="micro")*100/cv
            precision_soe_KNN+=precision_score_KNN*100/cv
            recall_soe_KNN+=recall_score_KNN*100/cv
            
        print("KNN test accuracy : ",accuracy_score_KNN )
        print("KNN test précision : ", precision_mean_score_KNN)
        print("KNN test précision par classe: ",precision_soe_KNN )
        print("KNN test recall par classe: ",recall_soe_KNN)
        KNN_time=end - start
        print("KNN time en secondes: ",KNN_time)
        return accuracy_score_KNN,precision_mean_score_KNN,precision_soe_KNN,recall_soe_KNN

def SVM(X,Y,cv,tx,nbreClasse):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tx/100)
    if(cv==1):
        param_grid = {'C': [ 1000, 10000], 'gamma': [0.01, 0.001], 'kernel': ['rbf']}
        svm_clf = SVC()
        grid_search = GridSearchCV(svm_clf, param_grid, cv=10, scoring='neg_mean_squared_error')


        grid_search.fit(X_train, Y_train.values.ravel())
        clf4 = grid_search.best_estimator_
        start = time. time()
        y_pred_SVM = clf4.predict(X_test)
        end = time. time()
        SVM_time=end - start
        accuracy_score_SVM=accuracy_score(Y_test, y_pred_SVM)*100
        precision_score_SVM= precision_score(Y_test, y_pred_SVM, average=None)*100
        recall_score_SVM=recall_score(Y_test, y_pred_SVM, average=None)*100
        precision_mean_score_SVM=precision_score(Y_test, y_pred_SVM, average="micro")*100
        print("SVM test accuracy: ", accuracy_score_SVM)
        print("SVM test précision : ", precision_mean_score_SVM)
        print("\SVM test précision par classe: ",precision_score_SVM)
        print("\nSVM test recall par classe: ", recall_score_SVM)
        print("\n SVM le temps de traitement: ",SVM_time)
        return accuracy_score_SVM,precision_mean_score_SVM,precision_score_SVM,recall_score_SVM
    else:
        sss = StratifiedShuffleSplit(n_splits=cv, random_state=0)
        precision_soe_SVM=[0]*nbreClasse
        recall_soe_SVM=[0]*nbreClasse
        accuracy_score_SVM=0
        precision_mean_score_SVM=0
        start = time. time()

        for train_index, test_index in sss.split(X,Y):
            X_train_sc, X_test_sc = X.iloc[train_index], X.iloc[test_index]
            Y_train_en, Y_test_en = Y.iloc[train_index], Y.iloc[test_index]

            param_grid = {'C': [ 1000, 10000], 'gamma': [0.01, 0.001], 'kernel': ['rbf']}
            svm_clf = SVC()
            grid_search = GridSearchCV(svm_clf, param_grid, cv=10, scoring='neg_mean_squared_error')

            grid_search.fit(X_train_sc, Y_train_en.values.ravel())
            clf4 = grid_search.best_estimator_
            y_pred_SVM = clf4.predict(X_test_sc)
            end = time. time()
            accuracy_score_SVM+=accuracy_score(Y_test_en, y_pred_SVM)*100/cv
            precision_score_SVM= precision_score(Y_test_en, y_pred_SVM, average=None)
            recall_score_SVM=recall_score(Y_test_en, y_pred_SVM, average=None)

            precision_mean_score_SVM+=precision_score(Y_test_en, y_pred_SVM, average="micro")*100/cv
            precision_soe_SVM+=precision_score_SVM*100/cv
            recall_soe_SVM+=recall_score_SVM*100/cv
            
        print("\n SVM test accuracy : ",accuracy_score_SVM )
        print("\n SVM test précision : ", precision_mean_score_SVM)
        print("\n SVM test précision par classe: ",precision_soe_SVM )
        print("\n SVM test recall par classe: ",recall_soe_SVM )
        DT_time=end - start
        print("\n SVM time en secondes: ",DT_time)
        return accuracy_score_SVM,precision_mean_score_SVM,precision_soe_SVM,recall_soe_SVM

def GB(X,Y,cv,tx,nbreClasse):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tx/100)
    if(cv==1):
        gnb_clf = GaussianNB()
        param_grid = {}
        grid_search = GridSearchCV(gnb_clf, param_grid,cv=5, scoring='neg_mean_squared_error')


        grid_search.fit(X_train, Y_train.values.ravel())
        clf4 = grid_search.best_estimator_
        start = time. time()
        y_pred_GB = clf4.predict(X_test)
        end = time. time()
        GB_time=end - start
        accuracy_score_GB=accuracy_score(Y_test, y_pred_GB)*100
        precision_score_GB= precision_score(Y_test, y_pred_GB, average=None)*100
        recall_score_GB=recall_score(Y_test, y_pred_GB, average=None)*100
        precision_mean_score_GB=precision_score(Y_test, y_pred_GB, average="micro")*100
        print("GB test accuracy: ", accuracy_score_GB)
        print("GB test précision : ", precision_mean_score_GB)
        print("\nGB test précision par classe: ",precision_score_GB)
        print("\nGB test recall par classe: ", recall_score_GB)
        print("\n GB le temps de traitement: ",GB_time)
        return accuracy_score_GB,precision_mean_score_GB,precision_score_GB,recall_score_GB
    else:
        sss = StratifiedShuffleSplit(n_splits=cv, random_state=0)
        precision_soe_GB=[0]*nbreClasse
        recall_soe_GB=[0]*nbreClasse
        accuracy_score_GB=0
        precision_mean_score_GB=0
        start = time. time()

        for train_index, test_index in sss.split(X,Y):
            X_train_sc, X_test_sc = X.iloc[train_index], X.iloc[test_index]
            Y_train_en, Y_test_en = Y.iloc[train_index], Y.iloc[test_index]
            gnb_clf = GaussianNB()
            param_grid = {}
            grid_search = GridSearchCV(gnb_clf, param_grid,cv=5, scoring='neg_mean_squared_error')


            grid_search.fit(X_train_sc, Y_train_en.values.ravel())
            clf4 = grid_search.best_estimator_
            y_pred_GB = clf4.predict(X_test_sc)
            end = time. time()
            accuracy_score_GB+=accuracy_score(Y_test_en, y_pred_GB)*100/cv
            precision_score_GB= precision_score(Y_test_en, y_pred_GB, average=None)
            recall_score_GB=recall_score(Y_test_en, y_pred_GB, average=None)

            precision_mean_score_GB+=precision_score(Y_test_en, y_pred_GB, average="micro")*100/cv
            precision_soe_GB+=precision_score_GB*100/cv
            recall_soe_GB+=recall_score_GB*100/cv
            
        print("GB test accuracy : ",accuracy_score_GB )
        print("GB test précision : ", precision_mean_score_GB)
        print("GB test précision par classe: ",precision_soe_GB )
        print("GB test recall par classe: ",recall_soe_GB )
        GB_time=end - start
        print("GB time en secondes: ",GB_time)
        return accuracy_score_GB,precision_mean_score_GB,precision_soe_GB,recall_soe_GB

def RN(X,Y,cv,tx,nbreClasse):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tx/100)

    i=X.shape[1]
    modelMul = keras.Sequential([
   # keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=i,input_dim=i, activation="softmax"),
    keras.layers.Dense(units=nbreClasse, activation="softmax"),
    #keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    modelMul.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    modelMul.fit(X_train, Y_train, epochs=cv,batch_size=10)
    predMul50 = modelMul.predict_classes(X_test)
    print(metrics.accuracy_score(Y_test,predMul50))