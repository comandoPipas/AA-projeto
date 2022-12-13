"""
Aprendizagem Automatica -- Projeto Final

M12816 - Cristiano Santos
E10973 - Sara Martins
"""

""" BIBLIOTECAS """

import datatest
import matplotlib as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import warnings
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

""" VARIAVEIS """

colunas = {'id', 'label', 'text', 'label_num'}

""" FUNCOES """

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
    # filtrar os avisos para que os de patern matching não sejam impressos
    warnings.filterwarnings("ignore")

    # carregar a funcao stopwords da biblioteca nltk
    palavras_stop = set(stopwords.words('english'))

    # atravessar cada entrada do dataset para executar o pre-processamento do texto
    for indice, linha in email.iterrows():
        if type(linha['text']) is str:
            texto = ""
            # substituir cada caratere especial com espacos
            linha['text'] = re.sub('[^a-zA-Z0-9\n]', ' ', linha['text'])
            # substituir espacos multiplos por um espaco unico
            linha['text'] = re.sub('\s+', ' ', linha['text'])
            # converter todos os carateres para minusculos 
            linha['text'] = linha['text'].lower()
            for palavra in linha['text'].split():
                # se a palavra nao e uma stop word, entao e mantida
                if not palavra in palavras_stop:
                    texto = texto + palavra + " "
            email['text', indice] = texto
        else:
            # o else e para programacao defensiva, para a eventualidade de
            # haver valores nulos na coluna de conteudo textual
            print ("Nao ha descricao textual para o indice ", indice, ".")

    return email

# funcao que abre e valida o dataset
# input: null
# output: dataframe
def preparar_dataframe():
    # abrir o csv com o dataset
    email = pd.read_csv('spam_ham_dataset.csv')

    # verificar a validação do dataset
    datatest.validate(email.columns, colunas)

    # uniformizar a formatacao das strings de texto
    email = reformatar(email)

    return email

# funcao que divide o dataset em conjuntos de treino e teste
# input: dataframe
# output: array, array, array, array
def dividir_dataframe(email):
    # escolher apenas as colunas de relevo para o problema
    email_dataframe = pd.DataFrame({'text': email['text'], 'spam/ham': email['label_num']})
    x = email_dataframe['text']
    y = email_dataframe['spam/ham']

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
def plot_matriz_confusao(y_teste, y_previsto):
    # declarar a matriz de confusao
    c = confusion_matrix(y_teste, y_previsto)
    labels = [0, 1]

    # representar a matriz de confusao num formato heatmap
    print("-" * 40, "Matriz de Confusao", "-" * 40)
    plt.figure(figsize = (8, 6))
    sns.heatmap(c, annot = True, cmap = "YlGnBu", fmt = ".3f", xticklabels = labels, yticklabels = labels)
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Original')
    plt.show()

# inicializacao da execucao do codigo
if __name__ == "__main__":
    email = preparar_dataframe()
    # verificar_estatisticas(email)
    x_treino, x_teste, y_treino, y_teste = dividir_dataframe(email)
    texto_treino, texto_teste = fatorizar_texto(x_treino, x_teste)



### IMPORTANTE --- PARA O RELATORIO
# 1. Não se faz normalização de dados porque, como o texto é constituído por strings,
# o facto de utilizar one-hot-encoding para tornar estas strings em valores numericos
# iria implicar perda de dados e dados categoricos nao podem ser normalizados
# 2. Como cada entrada string está desformatada, é necessário executar uma reformatação
# do texto. Este seria outro problema de um dataset em português: esta reformatação não
# poderia ser feita de forma tão linear e implicaria a troca manual de carateres
# acentuados ou a total ausencia destes
# 3. Documentação dos warnings: https://docs.python.org/3/library/warnings.html
# 4. Documentação dos NLTK: https://pythonspot.com/nltk-stop-words/
# 5. Deixa de ser burra e vai ver como se fazem for's outra vez
# 6. TFIDF é extremamente importante. ver!!!
# 6.1. min_dffloat or int, default=1 --- When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# 6.2. max_featuresint, default=None --- If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
