# DISCLAIMER

# Aprendizagem Automatica -- Projeto Final
# M12816 - Cristiano Miguel Abrantes Santos
# E10973 - Sara Maria da Silva Martins


# BIBLIOTECAS

import datatest
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import warnings
from nltk.corpus import stopwords
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


# VARIAVEIS

colunas = {"id", "label", "text", "label_num"}
# hinge - svm ;; log_loss - regressao logistica ;; perceptron - rede neuronal
algoritmos = [
    KNeighborsClassifier(),
    MultinomialNB(),
    SGDClassifier(alpha = 1, loss = "hinge"),
    SGDClassifier(alpha = 1, loss = "log_loss"),
    SGDClassifier(alpha = 1, loss = "perceptron"),
    ]


# FUNCOES

# funcao que mostra caracteristicas relativas ao dataset
# input: dataframe
# output: null
def verificar_estatisticas(email):
    # mostrar as primeiras cinco entradas do dataset
    print (email.head())
    # obter uma descricao estatistica do dataset
    print (email.describe())
    # mostrar o par (linhas, colunas) com os valores totais
    print (email.shape)

# funcao que uniformiza as strings de texto de email
# input: dataframe
# output: dataframe
def reformatar(email):
    # filtrar os avisos para que os de pattern matching não sejam impressos
    warnings.filterwarnings("ignore")

    # carregar a funcao stopwords da biblioteca nltk
    palavras_stop = set(stopwords.words("english"))
    
    # atravessar cada entrada do dataset para executar o pre-processamento do texto
    for indice, linha in email.iterrows():
        if type(linha["text"]) is str:
            texto = ""
            # substituir cada caratere especial com espacos
            linha["text"] = re.sub("[^a-zA-Z0-9\n]", " ", linha["text"])
            # substituir espacos multiplos por um espaco unico
            linha["text"] = re.sub("\s+", " ", linha["text"])
            # converter todos os carateres para minusculos 
            linha["text"] = linha["text"].lower()
            for palavra in linha["text"].split():
                # se a palavra nao e uma stop word, entao e mantida
                if not palavra in palavras_stop:
                    texto = texto + palavra + " "
            email["text", indice] = texto
        else:
            # o else e para programacao defensiva, para a eventualidade de
            # haver valores nulos na coluna de conteudo textual
            print ("Nao ha descricao textual para o indice ", indice, ".")

    # retornar o dataframe com a reformatacao de texto
    return email

# funcao que abre e valida o dataset
# input: null
# output: dataframe
def preparar_dataframe():
    # abrir o csv com o dataset
    email = pd.read_csv("spam_ham_dataset.csv")

    # verificar a validação do dataset
    datatest.validate(email.columns, colunas)

    # uniformizar a formatacao das strings de texto
    email = reformatar(email)

    # retornar o dataframe pronto para treino
    return email

# funcao que divide o dataset em conjuntos de treino e teste
# input: dataframe
# output: array, array, array, array
def dividir_dataframe(email):
    # escolher apenas as colunas de relevo para o problema
    email_dataframe = pd.DataFrame({"text": email["text"], "spam/ham": email["label_num"]})
    x = email_dataframe["text"]
    y = email_dataframe["spam/ham"]

    # dividir os dados
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = 0)

    # executar validacao cruzada
    x_treino, x_teste, y_treino, y_teste = train_test_split(x_treino, y_treino, test_size = 0.3, stratify = y_treino, random_state = 0)

    # devolver os diferentes conjuntos
    return x_treino, x_teste, y_treino, y_teste

# funcao para a fatorizacao tfidf dos dados de texto
# input: array, array
# output: matrix(n_samples, n_features), matrix(n_samples, n_features)
def fatorizar_texto(x_treino, x_teste):
    # inicializar o vetorizador de TFIDF
    vetorizador = TfidfVectorizer(min_df = 10, max_features = 5000)

    # reconhecer vocabulario e idf do conjunto de treino
    vetorizador.fit(x_treino.values)

    # transformar documentos/array em matriz
    texto_treino = vetorizador.transform(x_treino.values)
    texto_teste = vetorizador.transform(x_teste.values)

    # devolver os diferentes conjuntos
    return texto_treino, texto_teste

# funcao para o plot da matriz de confusao
# input: array, array
# output: null
def plot_matriz_confusao(algoritmo, y_teste, y_previsto):
    # declarar a matriz de confusao
    c = confusion_matrix(y_teste, y_previsto)
    labels = [0, 1]

    # representar a matriz de confusao num formato heatmap
    plt.figure(figsize = (8, 6))
    sns.heatmap(c, annot = True, cmap = "YlGnBu", fmt = ".3f", xticklabels = labels, yticklabels = labels)
    plt.title(algoritmo)
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Original")

    # mostrar a matriz de confusao
    plt.show()

# funcao para o treino do dataset com varios algoritmos de aprendizagem: 
# input: matrix(n_samples, n_features), array, matrix(n_samples, n_features), array
# output: null
def treino(texto_treino, y_treino, texto_teste, y_teste):
    for a in algoritmos:
        # guardar o nome do algoritmo para a precisao
        nome = str(a).split("(")[0]
        # efetuar o treino sobre o classificador
        classificador = a.fit(texto_treino, y_treino)
        # inicializar um classificador com calibracao de probabilidades com regressao logistica
        classificador_calibrado = CalibratedClassifierCV(classificador)
        classificador_calibrado.fit(texto_treino, y_treino)
        # efetuar a previsao sobre os resultados treinados
        y_previsto = classificador_calibrado.predict(texto_teste)
        # imprimir a precisao do algoritmo
        print("Precisao", nome, ": ", accuracy_score(y_teste, y_previsto))
        # representar a matriz de confusao
        plot_matriz_confusao(nome, y_teste, y_previsto)

# inicializacao da execucao do codigo
if __name__ == "__main__":
    email = preparar_dataframe()
    # verificar_estatisticas(email)
    x_treino, x_teste, y_treino, y_teste = dividir_dataframe(email)
    texto_treino, texto_teste = fatorizar_texto(x_treino, x_teste)
    treino(texto_treino, y_treino, texto_teste, y_teste)