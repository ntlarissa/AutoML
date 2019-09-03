from visualisation import*
from preprocessing import*
from featureengineering import*
from learningmodel import*
from evaluationmetric import*

def Choix(i):
    if i ==1 :
        return InitVisual()
    elif i == 2:
        return InitPrepro()
    elif i == 3:
        return InitFeaEng()
    elif i == 4:
        return InitLeaMod()
    elif i == 5:
        return InitEvaMod()
    elif i == 6:
        return Quit()
    else:
        return "Veuillez choisir la bonne opération!!!"

def IsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def Process():
    OperationList()
    nbre = input("\nEntrez le nombre correspondant à votre opération :")
    print("\n")
    if(IsInt(nbre)):
        q=Choix(int(nbre)).lower()
        if q=='y':
            Process()
        elif q=='n':
            print("\nMerci d\'avoir utilisé notre outil!!!")
        else:
            print("\nVeuillez entrer la bonne lettre!!!")
            Process()
    else:
        print("Veuillez entrer un chiffre!!!")
        Process()

def OperationList():
    print('\n')
    print('         Quelles opérations voulez vous effectuer???\n              1 - Visualisation des features\n              2 - Prétraitements des données')
    print('              3 - Réduction de dimensions\n              4 - Apprentissage du modèle\n              5 - Evaluation des performances\n              6 - Quitter l\'outil')


print('**** Bienvenue dans l\'assistance  d\' apprentissage ****')
Process()