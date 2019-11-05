"""
                            ----------------------------------------------------------
                                    Created on Sun Oct 27 14:13:25 2019
                                                @author: dp
                                    Nome: Danilo Pereira de Oliveira
                                        Matrícula: 31721BSI005
                                        Disciplina: Mineração de Dados
                                    Prof. Dr: Marcos Luiz de Paula Bueno
                                    Universidade Federal de Uberlândia
                            ----------------------------------------------------------
"""

import pandas as pd
import numpy as np  #Biblioteca para computação ciêntifica
import math #Biblioteca para funções matemáticas
import matplotlib.pyplot as plt  #biblioteca para visualização de dados
from sklearn import dummy  #classificador para fazer previsões usando regras simples
from sklearn import model_selection   #Dividir matrizes em subconjuntos aleatórios de train e test
from sklearn.model_selection import KFold  #Fornece índices de treinamento / teste para dividir dados em conjuntos de treinamento / teste
from sklearn import neighbors, metrics   #Classificador que implementa o voto dos k-vizinhos mais próximos.
from sklearn import preprocessing   #pacote fornece várias funções utilitárias comuns e classes de transformadores para alterar 
#vetores de recursos brutos em uma representação mais adequada para os estimadores a jusante.
import warnings
warnings.filterwarnings(action='ignore')

seed = 42  #gerador de números pseudo-aleatórios

data = pd.read_csv('winequality-white.csv', sep=";")   #preparação dos dados
data.head()

#Nossos dados contêm 12 colunas, 11 que correspondem a vários indicadores físico-químicos
# e 1 que é a qualidade do vinho.

#Extrairemos duas matrizes numpy desses dados, uma contendo os pontos e a outra contendo as tags:

X = data.as_matrix(data.columns[:-1])
y_m = data.as_matrix([data.columns[-1]])
y = y_m.flatten()

#Agora podemos exibir um histograma para cada uma de nossas variáveis:
fig = plt.figure(figsize=(16, 12))
for feat_idx in range(X.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X[:, feat_idx], bins=50, color='steelblue',
                normed=True, edgecolor='none')
    ax.set_title(data.columns[feat_idx], fontsize=14)

ax = fig.add_subplot(3,4, 12)
h = ax.hist(y, bins=50, color='steelblue',
            normed=True, edgecolor='none')
ax.set_title('Quality', fontsize=14)
plt.show()

#Em particular, essas variáveis ​​recebem valores em conjuntos diferentes.
# Por exemplo, "sulfatos" varia de 0 a 1, enquanto "dióxido de enxofre total" varia de 0 a 440. 
#Portanto, será necessário padronizar os dados para que o segundo não domine completamente o primeiro.

#Começaremos por transformar esse problema em um problema de classificação: 
#será uma questão de separar os bons vinhos dos medíocres:

y_class = np.where(y<6, 0, 1)

#Vamos dividir nossos dados em treinamento e teste. O conjunto de testes conterá 10% dos dados.
# Definimos 'random_state' com o valor da semente para que a separação dos dados seja fixa e reproduzível.
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y_class,
                                    test_size=0.1, # 10% dos dados no conjunto de teste
                                     random_state = seed
                                    )

def dist_euclidiana(v1, v2):
	v1, v2 = np.array(v1), np.array(v2)    
	diff = v1 - v2
	quad_dist = np.dot(diff, diff)
	return math.sqrt(quad_dist)

n1,n2 = np.shape(X_test)

for lin in X_test[:490]:
    saida = []
    new = open("saida-teste.txt","w")
    for lin1 in X_train:
        saida.append((dist_euclidiana(lin,lin1),lin1[10]))
    saida.sort()
    for ii in saida:
        new.writelines(str(ii[0])+" "+str(ii[1])+"\n")    
    new.close()
      
#Agora podemos padronizar os dados de treinamento e aplicar a mesma transformação aos dados de teste:
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

#Os dados podem ser visualizados novamente para garantir que as diferentes variáveis ​​recebam valores 
#que agora possuem ordens de magnitude semelhantes.
fig = plt.figure(figsize=(16, 12))
for feat_idx in range(X_train_std.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X_train_std[:, feat_idx], bins=50, color='steelblue',
        normed=True, edgecolor='none')
    ax.set_title(data.columns[feat_idx], fontsize=14)
    
# Criamos a dobra de 5 k usando a semente para que essas dobras sejam reproduzíveis:
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X_train_std):
    print("TRAIN:", "taille", len(train_index), ", 5ers indices", train_index[0:5], 
          "|| TEST:", "taille", len(test_index), ", 5ers indices", test_index[0:5])
    
#Agora usaremos o método "GridSearchCV" para validar cruzadamente o parâmetro k 
#de um kNN (o número de vizinhos mais próximos) no jogo de treinamento:

# Defina os valores dos hiperparâmetros a serem testados
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}
# Escolha uma pontuação para otimizar, aqui a precisão (proporção de previsões corretas)
score = 'accuracy'

# Cria um classificador kNN com pesquisa de hiperparâmetro de validação cruzada
clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), # um classificador kNN
                                    param_grid, # hiperparâmetros para testar
                                    cv=kf, # dobras para validação cruzada
                                    scoring=score # pontuação para otimizar
                                    )
# Otimiza esse classificador no treinamento
clf.fit(X_train_std, y_train)

# Visualizar o(s) hiperparâmetro(s) ideal(is)
print("Melhores hiperparâmetros na base de treinamento: ")
print(clf.best_params_)

#Ver o desempenho correspondente
print("Resultados da validação cruzada: ")
for mean, std, params in zip(clf.cv_results_['mean_test_score'], # pontuação média
                                clf.cv_results_['std_test_score'], # desvio padrão da pontuação
                                clf.cv_results_['params'] # valor do hiperparâmetro
                                ):
    print("\t%s = %0.3f (+/-%0.03f) for %r" % (score, # critério usado
                                                mean, # pontuação média
                                                std * 2, # barra de erro
                                                params # hiperparâmetro
                                                ))
    
pd.DataFrame(clf.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]

#O melhor desempenho (~ 0,771) é alcançado aqui com 3 vizinhos.

#Agora podemos assistir ao desempenho no teste. 
#O GridSearchCV treinou automaticamente o melhor modelo em todo o jogo de treinamento.
y_pred = clf.predict(X_test_std)
print("\nPontuação de precisão base de teste: %0.3f" % metrics.accuracy_score(y_test, y_pred))

#Começa criando a função que executará a validação cruzada:
def cvf(X, y, n_neighbors, n_fold, verbose=False):
    """
    X : dados
    y : valor a prever
    n_neighbors : lista de K para testar o K-NN
    n_fold : número de dobras a serem criadas
    verbose : Falso por padrão, exibe informações de código
    """
    
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    if verbose: 
        for train_index, test_index in kf.split(X):
            print("TRAIN:", "taille", len(train_index), ", 5ers indices", train_index[0:5], 
                  "|| TEST:", "taille", len(test_index), ", 5ers indices", test_index[0:5])
    
    # inicializa a tabela que conterá os diferentes resultados
    accuracy_mean = [] 
    
    # faz um loop para testar os diferentes valores de K para o KNN
    for kn in n_neighbors:
        if verbose: print("kn", kn)    
        knn = neighbors.KNeighborsClassifier(kn) # inicialização do classificador
        
        accuracy = [] # inicialização de uma lista que conterá a precisão de cada dobra k e cada knn
        
        # faz um loop nas diferentes dobras
        for train_index, test_index in kf.split(X):
            if verbose: print("TRAIN:", "taille", len(train_index), ", 5ers indices", train_index[0:5], 
                      "|| TEST:", "taille", len(test_index), ", 5ers indices", test_index[0:5])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
                        
            # fiz o knn na dobra de treinamento
            knn.fit(X_train, y_train)
            
            # testa a precisão na dobra de teste e a adiciona à tabela 'precisão':
            accuracy.append([knn.score(X_test, y_test)])
        if verbose: print(np.mean(accuracy))
        
        # os resultados são agregados mantendo apenas a média da precisão das diferentes dobras para cada k
        accuracy_mean.append([kn, round(np.mean(accuracy), 3)])
    
    return accuracy_mean

#Gira a função com os mesmos valores usados ​​para o GridSearchCV:
n_neighbors = [3, 5, 7, 9, 11, 13, 15]
n_fold = 5
cvf = cvf(X_train_std, y_train, n_neighbors, n_fold, verbose=False)
# A tabela de precisão média para cada K é obtida:
cvf_df = pd.DataFrame(cvf, columns=['knn', 'mean_accuracy'])
cvf_df
cvf_df[cvf_df['mean_accuracy'] == cvf_df['mean_accuracy'].max()]
#O K a ser selecionado é, portanto, 3, com uma precisão média de 0,771, 
#o que corresponde ao que o GridSearchCV retornou com os mesmos dados.
#Agora podemos verificar o desempenho da base de teste.
knn5 = neighbors.KNeighborsClassifier(3)
knn5.fit(X_train_std, y_train)
print("\nAccurácia na base de teste: %0.3f" % knn5.score(X_test_std, y_test))

#Encontramos a mesma precisão com o GridSearchCV.

#Avaliação da classificação knn
y_pred_proba = clf.predict_proba(X_test_std)[:, 1]
[fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='coral', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite', fontsize=14)
plt.ylabel('Sensibilite', fontsize=14)
plt.show()

print("AUROC =", round(metrics.auc(fpr, tpr), 3))

#Veja o exemplo do vinho verde. 
#Imagine que o algoritmo deve ser capaz de detectar com eficácia vinhos de baixa qualidade, 
#que não serão examinados por um especialista humano. 
#Em seguida, queremos limitar o número de falsos negativos, 
#para limitar o número de liberações infundadas. 
#Vamos definir uma taxa de falsos negativos toleráveis ​​(a proporção de positivos negativos previstos incorretamente) de 5%.
# Isso equivale a uma sensibilidade de 0,95:

idx = np.min(np.where(tpr > 0.95)) # primeiro índice limite para o qual
                                   # a sensibilidade é maior que 0,95
print("Sensibilidade : %.2f" % tpr[idx])
print("Especificidade : %.2f" % (1-fpr[idx]))
print("Seuil : %.2f" % thr[idx])

#Usar um limite de 0,20 garante uma sensibilidade de 0,98 e uma especificidade de 0,18, 
#uma taxa de falsos positivos de ... 82%.

#Avalie comparando com abordagens ingênuas
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y,
                                    test_size=0.1, # 10% dos dados no conjunto de teste
                                     random_state = seed
                                    )
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

#Vamos treinar um kNN com k = 11 nesses dados:
knn2 = neighbors.KNeighborsRegressor(n_neighbors=11)

knn2.fit(X_train_std, y_train)

#E aplique-o para prever os rótulos do nosso jogo de teste:
y_pred = knn2.predict(X_test_std)   

print("RMSE : %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Valores previstos VS valores reais:
plt.scatter(y_test, y_pred, color='green')
plt.axis([2, 9, 2, 9])
plt.show()

#Como nossos rótulos assumem valores inteiros entre 3 e 8, 
#temos muitos pontos sobrepostos nas mesmas coordenadas. 
#Para visualizar melhor os dados, podemos usar como círculos marcadores cujo tamanho é proporcional 
#ao número de pontos presentes nessas coordenadas.
sizes = {} # chave: coordenadas; valor: número de pontos nessas coordenadas
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes:
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter([k[0] for k in keys], # valor verdadeiro (abscissa)
            [k[1] for k in keys], # valor previsto (pedido)
            s=[sizes[k] for k in keys], # tamanho do marcador
            color='green')      
plt.show()

#Notamos, portanto, um acúmulo de previsões corretas na diagonal.
#No entanto, o modelo não é muito preciso em suas previsões.

#Para entender melhor nosso modelo, vamos compará-lo a uma primeira abordagem ingênua, 
#que é prever valores aleatórios, distribuídos uniformemente entre os valores alto e baixo 
#das tags do conjunto de dados de treinamento.
y_pred_random = np.random.randint(np.min(y), np.max(y), y_test.shape)

#Calcule o RMSE correspondente:
print("RMSE : %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred_random)))

#Obtemos um RMSE muito superior ao RMSE obtido pelo nosso modelo kNN. 
#Nosso modelo conseguiu assim aprender muito melhor do que um modelo aleatório.

#No entanto, muitos de nossos vinhos têm uma classificação de 6, e muitas de nossas previsões estão em torno desse valor. 


