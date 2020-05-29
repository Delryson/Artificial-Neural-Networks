#Delryson Saraiva
#Rede Perceptron simples para classificação

# Livro: Redes Neurais Artificiais - Ivan N. da Silva
# Projeto Prático 3.6; pag. 70; Perceptron

import random

learnRate = 0.01 #Taxa de aprendizagem fornecida no livro

#Conjunto de Treinamento com os valores x1, x2 e x3 de cada amostra:
x = [[-0.6508,0.1097,4.0009],
     [-1.4492,0.8896,4.4005],
     [2.0850,0.6876,12.0710],
     [0.2626,1.1476,7.7985],
     [0.6418,1.0234,7.0427],
     [0.2569,0.6730,8.3265],
     [1.1155,0.6043,7.4446],
     [0.0914,0.3399,7.0677],
     [0.0121,0.5256,4.6316],
     [-0.0429,0.4660,5.4323],
     [0.4340,0.6870,8.2287],
     [0.2735,1.0287,7.1934],
     [0.4839,0.4851,7.4850],
     [0.4089,-0.1267,5.5019],
     [1.4391,0.1614,8.5843],
     [-0.9115,-0.1973,2.1962],
     [0.3654,1.0475,7.4858],
     [0.2144,0.7515,7.1699],
     [0.2013,1.0014,6.5489],
     [0.6483,0.2183,5.8991],
     [-0.1147,0.2242,7.2435],
     [-0.7970,0.8795,3.8762],
     [-1.0625,0.6366,2.4707],
     [0.5307,0.1285,5.6883],
     [-1.2200,0.7777,1.7252],
     [0.3957,0.1076,5.6623],
     [-0.1013,0.5989,7.1812],
     [2.4482,0.9455,11.2095],
     [2.0149,0.6192,10.9263],
     [0.2012,0.2611,5.4631]]

#Resposta d esperada do conjunto de treinamento:
d = [-1.0000,-1.0000,-1.0000,1.0000,1.0000,
     -1.0000,1.0000,-1.0000,1.0000,1.0000,
     -1.0000,1.0000,-1.0000,-1.0000,-1.0000,
     -1.0000,1.0000,1.0000,1.0000,1.0000,
     -1.0000,1.0000,1.0000,1.0000,1.0000,
     -1.0000,-1.0000,1.0000,-1.0000,1.0000,]

#Contador de Épocas:
epoca = 0 #treinamento de cada amostra
total_epoca = 0 #total de todas as amostras


#Adicionando o BIAS = -1 nas amostras para ajustar o peso inicial
for list in x: #percorrendo as listas dentro das amostras
     list.insert(0,-1) #inserindo -1 na primeira posição
#Esse valor vai ajudar a ajustar o peso inicial.

#Vetor Peso inicial de cada amostra
initial_weight  = [random.random() for j in range(len(x[0]))] #Vetor Peso
#O vetor contém valores aleatórios entre 0 e 1 correspondente a cada variável de entrada
#para cada vetor de amostra. No caso desta rede, são 3 pesos para 3 entradas e 1 peso para o BIAS


### INÍCIO DO TREINAMENTO ###

def activation(y): #Função Degrau Bipolar
     if y >= 0:
          return 1
     if y < 0:
          return -1

def weighted_sum(amostra,peso):
     peso_ponderado = []
     for i in range(len(amostra)):
          a = amostra[i] * peso[i]
          peso_ponderado.append(a)
     return peso_ponderado

def adjustment(initial_weight,sample,expected_result,learnRate,y):
     new_weights = []
     for i in range(len(sample)):
          a = initial_weight[i] + learnRate*(expected_result-y)*sample[i]
          new_weights.append(a)
     return new_weights


for i in range(len(x)):
     sample = x[i]
     expected_result = d[i]
     weight = weighted_sum(sample,initial_weight)
     u = sum(weight)
     y = activation(u)

     while (y != expected_result):
          initial_weight = adjustment(initial_weight,sample,expected_result,learnRate,y)
          weight = weighted_sum(sample,initial_weight)
          u = sum(weight)
          y = activation(u)
          epoca += 1
          total_epoca += 1

     else:
          if y == 1:
               print("Classe A")
          else:
               print("Classe B")

          print ("Épocas:",epoca,'\n')
          epoca = 0


print ("Total de épocas: ", total_epoca)
print ("Peso encontrado no treinamento:",initial_weight)
### FIM TREINAMENTO ###
