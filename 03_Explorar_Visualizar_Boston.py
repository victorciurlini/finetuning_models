#==============================================================================
# EXPERIMENTO 05 - EXPLORANDO E VISUALIZANDO O CONJUNTO "BOSTON" (REGRESSÃO)
#==============================================================================

import pandas as pd
import math
import sys

from sklearn.metrics import mean_squared_error

from scipy.stats import pearsonr

#------------------------------------------------------------------------------
# Importar conjunto de dados de planilha Excel para dataframe Pandas
#------------------------------------------------------------------------------

dados = pd.read_excel("D02_Boston.xlsx")

#------------------------------------------------------------------------------
# Descartar a primeira coluna
#------------------------------------------------------------------------------

dados = dados.iloc[:,1:]

#------------------------------------------------------------------------------
# Verificar as colunas disponíveis
#------------------------------------------------------------------------------

colunas = dados.columns

print("Colunas disponíveis:")
print(colunas)

#------------------------------------------------------------------------------
# Plotar diagramas de dispersão entre cada atributo e o alvo
#------------------------------------------------------------------------------

# PARA VER OS GRÁFICOS, RETIRE O COMENTÁRIO "#" DAS DUAS LINHAS ABAIXO:

for col in colunas:
    dados.plot.scatter(x=col,y='target')

sys.exit()

#------------------------------------------------------------------------------
# Listar os coeficientes de Pearson entre cada atributo e o alvo
#------------------------------------------------------------------------------

for col in colunas:
    print('%10s = %6.3f' % ( col , pearsonr(dados[col],dados['target'])[0] ) )

#------------------------------------------------------------------------------
# Explorar correlações mútuas entre os atributos
#------------------------------------------------------------------------------

atributo1 = "RM"
atributo2 = "PTRATIO"

print('%10s = %6.3f' % (
    atributo1+'_'+atributo2,
    pearsonr(dados[atributo1],dados[atributo2])[0] ) )

# PARA VER OS GRÁFICO, RETIRE O COMENTÁRIO "#" DA LINHA ABAIXO:

#dados.plot.scatter(x=atributo1,y=atributo2)

#------------------------------------------------------------------------------
#  Selecionar os atributos utilizados no treinamento do modelo
#------------------------------------------------------------------------------

dados = dados[[
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT',
    'target'
    ]]

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dados.iloc[:, :-1].values
y = dados.iloc[:, -1].values.ravel()

#------------------------------------------------------------------------------
#  Aplicar uma escala a matriz X (METODOLOGICAMENTE INCORRETO, SÓ PRA ILUSTRAR)
#------------------------------------------------------------------------------

# from sklearn.preprocessing import StandardScaler

# scaler   = StandardScaler()
# X_scaled = scaler.fit_transform(X)


#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 200,
        random_state = 0
)

#------------------------------------------------------------------------------
#  Treinar um regressor kNN com o conjunto de treinamento
#------------------------------------------------------------------------------

K=1

from sklearn.neighbors import KNeighborsRegressor

modelo = KNeighborsRegressor( n_neighbors = K, weights='uniform' )
modelo = modelo.fit(X_train, y_train)
y_resposta = modelo.predict(X_test)

#------------------------------------------------------------------------------
#  Verificar a variação do erro com o parâmetro K (sem escala)
#------------------------------------------------------------------------------

print ( ' ' )
print ( ' K-NN SEM ESCALA:' )
print ( ' ' )
print ( '   K      Erro' )
print ( ' ----     -------' )

for k in range(1,21):

    knn = KNeighborsRegressor(
            n_neighbors=k,
            weights='uniform'
            )

    knn = knn.fit(X_train, y_train)

    y_resposta  = knn.predict(X_test)

    erro = math.sqrt ( mean_squared_error ( y_test  , y_resposta  ) )

    print ( str ( '   %2d' % k    ) + '  ' +
            str ( '%10.4f' % erro )
          )


#------------------------------------------------------------------------------
#  Verificar a variação do erro com o parâmetro K (com escala)
#------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler  = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print ( ' ' )
print ( ' K-NN COM ESCALA:' )
print ( ' ' )
print ( '   K      Erro' )
print ( ' ----     -------' )

for k in range(1,21):

    knn = KNeighborsRegressor(
            n_neighbors=k,
            weights='distance'
            )

    knn = knn.fit(X_train, y_train)

    y_resposta  = knn.predict(X_test)

    erro = math.sqrt ( mean_squared_error ( y_test  , y_resposta  ) )

    print ( str ( '   %2d' % k    ) + '  ' +
            str ( '%10.4f' % erro )
          )

#------------------------------------------------------------------------------
#  Treinar um regressor linear com o conjunto de treinamento
#------------------------------------------------------------------------------

print ( ' ' )
print ( ' REGRESSOR LINEAR:' )
print ( ' ' )

from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo = modelo.fit(X_train, y_train)

y_resposta  = modelo.predict(X_test)

erro = math.sqrt ( mean_squared_error ( y_test  , y_resposta  ) )

print ( str ( '   %2d' % k    ) + '  ' +
        str ( '%10.4f' % erro )
      )

#sys.exit()

#------------------------------------------------------------------------------
# Treinar e testar um regressor POLINOMIAL para graus de 1 a 5
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K:')
print(' ')

from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

print(' grau  n_atributos               RMSE               R2')
print(' ----  -----------  -----------------  ---------------')

for g in range(1,6):

    pf = PolynomialFeatures(degree=g)

    pf = pf.fit(X_train)
    X_train_poly = pf.transform(X_train)
    X_test_poly = pf.transform(X_test)

    regressor_linear = LinearRegression()

    regressor_linear = regressor_linear.fit(X_train_poly,y_train)

    y_resposta_teste  = regressor_linear.predict(X_test_poly)

    mse_out  = mean_squared_error(y_test,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_test,y_resposta_teste)

    print(' %4d  %10d  %17.4f  %17.4f' % ( g , X_train_poly.shape[1], rmse_out , r2_out ) )

print(' grau  n_atributos            RMSE IN         RMSE OUT')
print(' ----  -----------  -----------------  ---------------')

for g in range(1,10):

    pf = PolynomialFeatures(degree=g)

    pf = pf.fit(X_train)
    X_train_poly = pf.transform(X_train)
    X_test_poly = pf.transform(X_test)

    regressor_linear = LinearRegression()

    regressor_linear = regressor_linear.fit(X_train_poly,y_train)

    y_resposta_treino = regressor_linear.predict(X_train_poly)
    y_resposta_teste  = regressor_linear.predict(X_test_poly)

    mse_in  = mean_squared_error(y_train,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    mse_out  = mean_squared_error(y_test,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)

    print(' %4d  %10d  %17.4f  %17.4f' % ( g , X_train_poly.shape[1], rmse_in, rmse_out ) )

#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra em funcao do grau do polinomio
#------------------------------------------------------------------------------

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

print ( '           Regressao Simples   |  Regressao RIDGE (L2)  |  Regressao LASSO (L1)' )
print ( ' ' )
print ( ' Grau     Erro IN    Erro OUT  |   Erro IN    Erro OUT  |   Erro IN    Erro OUT' )
print ( ' ----     -------    --------  |   -------    --------  |   -------    --------' )

#modelo_ridge = Ridge ( alpha = np.power(10,-3.8))
#modelo_lasso = Lasso ( alpha = 1.E-9 )

modelo = LinearRegression()
modelo_ridge = Ridge ( alpha = np.power(10,-3.8) )
modelo_lasso = Lasso ( alpha = np.power(10,-3.7), max_iter=10000000 )

for degree in range(1,9):

    pf = PolynomialFeatures(degree)
    modelo = LinearRegression()

    # treinamento dos três modelos

    x_treino_poly = pf.fit_transform(x_treino)

    modelo = modelo.fit(x_treino_poly, y_treino)

    modelo_ridge = modelo_ridge.fit ( x_treino_poly , y_treino )
    modelo_lasso = modelo_lasso.fit ( x_treino_poly , y_treino )

    # predict dentro da amostra

    y_treino_pred = modelo.predict(x_treino_poly)

    y_treino_pred_ridge = modelo_ridge.predict(x_treino_poly)
    y_treino_pred_lasso = modelo_lasso.predict(x_treino_poly)

    # predict fora da amostra

    x_teste_poly = pf.transform(x_teste)

    y_teste_pred = modelo.predict(x_teste_poly)

    y_teste_pred_ridge = modelo_ridge.predict(x_teste_poly)
    y_teste_pred_lasso = modelo_lasso.predict(x_teste_poly)

    # calcular os erros

    RMSE_in  = math.sqrt ( mean_squared_error ( y_treino , y_treino_pred ) )
    RMSE_out = math.sqrt ( mean_squared_error ( y_teste  , y_teste_pred  ) )

    RMSE_in_ridge  = math.sqrt ( mean_squared_error ( y_treino , y_treino_pred_ridge ) )
    RMSE_out_ridge = math.sqrt ( mean_squared_error ( y_teste  , y_teste_pred_ridge  ) )

    RMSE_in_lasso  = math.sqrt ( mean_squared_error ( y_treino , y_treino_pred_lasso ) )
    RMSE_out_lasso = math.sqrt ( mean_squared_error ( y_teste  , y_teste_pred_lasso  ) )

    print ( str ( '   %2d' % degree   ) + '  ' +
            str ( '%10.4f' % RMSE_in  ) + '  ' +
            str ( '%10.4f' % RMSE_out ) + '  |' +
            str ( '%10.4f' % RMSE_in_ridge  ) + '  ' +
            str ( '%10.4f' % RMSE_out_ridge ) + '  |' +
            str ( '%10.4f' % RMSE_in_lasso  ) + '  ' +
            str ( '%10.4f' % RMSE_out_lasso )
          )
