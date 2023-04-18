import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def ler_dados(dataset_path):
    dados = pd.read_excel(dataset_path)

    dados = dados.iloc[:,1:]

    x = dados.iloc[:,:-1].values
    y = dados.iloc[:,-1].values

    return x, y, dados

def gera_base_treino_teste(dataset_path):

    x, y, dados = ler_dados(dataset_path)

    dados_embaralhados = dados.sample(frac=1,random_state=54321)

    x_treino = dados_embaralhados.iloc[:1297,:-1].values
    y_treino = dados_embaralhados.iloc[:1297,-1].values

    x_teste = dados_embaralhados.iloc[1297:,:-1].values
    y_teste = dados_embaralhados.iloc[1297:,-1].values

    return x_treino, x_teste, y_treino, y_teste


def avalia_parametros(modelo, param_grid, x_treino, y_treino):
    grid_search = GridSearchCV(modelo, param_grid, cv=5, return_train_score=True)
    
    # Ajustando o modelo com os dados de treinamento
    grid_search.fit(x_treino, y_treino)

    # Imprimindo os resultados da busca em grade
    print("Melhores parâmetros: {}".format(grid_search.best_params_))
    print("Melhor pontuação de validação cruzada: {:.2f}".format(grid_search.best_score_))
    k = grid_search.best_params_['n_neighbors']
    weights = grid_search.best_params_['weights']

    return grid_search.best_params_

def gera_tabela_metricas(y_teste, y_pred):
    accuracy = accuracy_score(y_teste, y_pred)
    # matriz_de_confusao = confusion_matrix(y_teste, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_teste, y_pred)
    df_report = pd.DataFrame({'precision': precision, 'recall': recall, 'fscore': fscore})

    return accuracy, df_report

def predict_models(models, lista_parametros, x_treino, y_treino, x_teste, y_teste):
    
    # List to store the predictions of each model
    lista_configuracao = []
    # Iterate over the models
    for model, hyperparameter in zip(models, lista_parametros):
        print("Novo modelo:")
        # Create a grid search object to find the best hyperparameters for the current model
        clf = GridSearchCV(model, hyperparameter, cv=5, return_train_score=True)

        # Fit the grid search object to the training data
        clf.fit(x_treino, y_treino)
        print("Melhores parâmetros: {}".format(clf.best_params_))
        print("Melhor pontuação de validação cruzada: {:.2f}".format(clf.best_score_))
        # Make predictions on the test data using the best model found by the grid search
        y_pred = clf.predict(x_teste)

        # Append the predictions to the list
        

        accuracy, df_report = gera_tabela_metricas(y_teste, y_pred)
        print(f"Acurácia: {accuracy}")
        # print(df_report)
        lista_configuracao.append({'parametros': clf.best_params_,
                                    'validacao_cruzada': clf.best_score_,
                                    'accuracy': accuracy,
                                    'dataframe': df_report})
    # Return the list of predictions
    return lista_configuracao

if __name__ == "__main__":
    dataset_path = 'datasets/Digits.xlsx'
    lista_modelos = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), SVC(), LogisticRegression()]
    lista_parametros = [
    {'n_neighbors': [i for i in range(1, 31)],
    'weights': ['uniform', 'distance']
    },
    {'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'min_samples_split': [2, 4, 6, 8, 10 ,12, 14, 16, 18, 20],
    'criterion': ['gini', 'entropy']
        },
    {'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
        },
    {'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
    },
    {'penalty': ['l2'],
    'C': [0.1, 1.0],
    'solver': ['newton-cg']
    }
    ]
    x_treino, x_teste, y_treino, y_teste = gera_base_treino_teste(dataset_path)
    lista_predicoes = predict_models(lista_modelos, lista_parametros, x_treino, y_treino, x_teste, y_teste)