#==============================================================================
# EXPERIMENTO 02 - CLASSIFICADOR KNN PARA O CONJUNTO DIGITOS
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#------------------------------------------------------------------------------
# Ler o planilha Excel com os dados do conjunto DIGITS
#------------------------------------------------------------------------------

dados = pd.read_excel('Digits.xlsx')

dados = dados.iloc[:,1:]

x = dados.iloc[:,:-1].values
y = dados.iloc[:,-1].values

#------------------------------------------------------------------------------
#  Visualizar alguns digitos
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

for i in range(0,10):
    plt.figure(figsize=(30,180))
    d_plot = plt.subplot(1, 10, i+1)
    d_plot.set_title("y = %.0f" % y[i])

    d_plot.imshow(x[i,:].reshape(8,8),
                  #interpolation='spline16',
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0 , vmax=16)
    #plt.text(-8, 3, "y = %.2f" % y[i])

    d_plot.set_xticks(())
    d_plot.set_yticks(())

plt.show()

#------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#------------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1,random_state=54321)

#------------------------------------------------------------------------------
# Criar os arrays X e Y para o conjunto de treino e para o conjunto de teste
#------------------------------------------------------------------------------

# conjunto de treino

x_treino = dados_embaralhados.iloc[:1297,:-1].values
y_treino = dados_embaralhados.iloc[:1297,-1].values

# conjunto de teste

x_teste = dados_embaralhados.iloc[1297:,:-1].values
y_teste = dados_embaralhados.iloc[1297:,-1].values

#-------------------------------------------------------------------------------
# Treinar um classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=3,weights='uniform')

classificador = classificador.fit(x_treino,y_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
#-------------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
#-------------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

#-------------------------------------------------------------------------------
# Verificar a acurácia do classificador
#-------------------------------------------------------------------------------

print ("\nCLASSIFICADOR KNN (DENTRO DA AMOSTRA)\n")

total   = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

print ("\nCLASSIFICADOR KNN (FORA DA AMOSTRA)\n")

total   = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros   = sum(y_resposta_teste!=y_teste)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

sys.exit()

matriz_de_confusao = confusion_matrix(y_resposta_teste,y_teste)
print('Matriz de Confusao:\n\n',matriz_de_confusao)



#-------------------------------------------------------------------------------
# Verificar os erros cometidos pelo classificador
#-------------------------------------------------------------------------------

indice_erro = np.where(y_resposta_teste != y_teste)[0]

for i in range(len(indice_erro)):
    plt.figure(figsize=(30,180))
    d_plot = plt.subplot(1, 10, i+1)
    d_plot.set_title("gabarito = %d ; resposta = %d" % (y_teste[indice_erro[i]],y_resposta_teste[indice_erro[i]]))

    d_plot.imshow(x_teste[indice_erro[i],:].reshape(8,8),
                  #interpolation='spline16',
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0 , vmax=16)
    #plt.text(-8, 3, "y = %.2f" % y[i])

    d_plot.set_xticks(())
    d_plot.set_yticks(())

plt.show()

#-------------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos
#-------------------------------------------------------------------------------

print ( "\n  K TREINO  TESTE")
print ( " -- ------ ------")

for k in range(1,31):

    classificador = KNeighborsClassifier(n_neighbors=k,weights='uniform')
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)

    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

    #acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
    #acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

    print(
        "%3d"%k,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )



#-------------------------------------------------------------------------------
# CLASSIFICADOR COM ÁRVORE DE DECISÃO
#-------------------------------------------------------------------------------

print ( ' ' )
print ( "CLASSIFICADOR COM ÁRVORE DE DECISÃO:")
print ( ' ' )

from sklearn.tree import DecisionTreeClassifier

classificador = DecisionTreeClassifier(
    criterion    = 'gini',   # 'gini' ou 'entropy'
    max_depth    = 16,
    random_state = 0
    )

classificador = classificador.fit(x_treino,y_treino)

y_resposta = classificador.predict(x_teste)

total   = len(y_teste)
acertos = sum(y_resposta==y_teste)
erros   = sum(y_resposta!=y_teste)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ( "Acurácia: %6.1f" % (100*acuracia) )


#-------------------------------------------------------------------------------
# CLASSIFICADOR RANDOM FOREST
#-------------------------------------------------------------------------------

print ( ' ' )
print ( "CLASSIFICADOR COM RANDOM FOREST:")
print ( ' ' )

from sklearn.ensemble import RandomForestClassifier

print ( "\n  N TREINO  TESTE")
print ( " -- ------ ------")

for k in range(5,201,5):

    classificador = RandomForestClassifier(
        n_estimators=k,
        random_state=123
        )

    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)

    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

    #acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
    #acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

    print(
        "%3d"%k,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )

#-------------------------------------------------------------------------------
# CLASSIFICADOR COM REGRESSÃO LOGÍSTICA
#-------------------------------------------------------------------------------

print ( ' ' )
print ( "CLASSIFICADOR COM REGRESSÃO LOGÍSTICA:")
print ( ' ' )

from sklearn.linear_model import LogisticRegression

print ( "\n  K TREINO  TESTE")
print ( " -- ------ ------")

print ( "\n    K TREINO  TESTE")
print ( " ---- ------ ------")

for k in range(-6,7):
#for C in [0.001,0.002,0.005,0.010,0.020,0.050,0.100]:

    C = 10**k

    classificador = LogisticRegression(penalty='l2', C=C, max_iter=100000)
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)

    acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
    acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

    print(
        "%3d"%k,
        #"%5.3f"%C,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )


#-------------------------------------------------------------------------------
# CLASSIFICADOR COM REDE NEURAL
#-------------------------------------------------------------------------------

import tensorflow as tf

# OS LABELS DEVEM SER BOOLEANOS, UM PARA CADA CLASSE

z_treino = np.zeros((y_treino.shape[0],10))
for i in range(y_treino.shape[0]):
    z_treino[i,y_treino[i]] = 1

z_teste  = np.zeros((y_teste.shape[0],10))
for i in range(y_teste.shape[0]):
    z_teste[i,y_teste[i]] = 1


print ( " ")
print ( " REDE NEURAL USANDO TENSORFLOW")
print ( " ")
print ( " NEURON  TREINO  TESTE")
print ( " ------ ------ ------")

for n in range(100,101,1):

    classificador = tf.keras.models.Sequential()

    # CAMADA DE ENTRADA COM 64 NEURONIOS (1 PARA CADA ATRIBUTO)

    classificador.add(tf.keras.layers.Flatten(input_shape=(64,)))

    # CAMADAS INTERNAS OCULTAS (PROJETO LIVRE)

    classificador.add(tf.keras.layers.Dense(100,tf.nn.sigmoid))
    classificador.add(tf.keras.layers.Dense(100,tf.nn.sigmoid))

    # CAMADA DE SAIDA COM 10 NEURONIOS (1 PARA CADA CLASSE)

    classificador.add(tf.keras.layers.Dense(10,tf.nn.sigmoid))

    classificador.compile (
        optimizer = "adam", # opcoes disponiveis: SGD RMSprop Adam Adadelta Adagrad Adamax Nadam Ftrl
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"]
        ) # vejam https://keras.io/api/models/model_training_apis/

    classificador.fit (
        x_treino,
        z_treino,
        validation_data = ( x_teste, z_teste ),
        batch_size = 32,
        epochs = 100,
        verbose = 1
        )

    z_resposta_treino = classificador.predict(x_treino)
    y_resposta_treino = np.argmax(z_resposta_treino,axis=1)

    z_resposta_teste  = classificador.predict(x_teste)
    y_resposta_teste  = np.argmax(z_resposta_teste,axis=1)

    acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
    acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

    print(
        "%6d"    % n,
        "%6.1f"  % (100*acuracia_treino),
        "%6.1f"  % (100*acuracia_teste)
        )













