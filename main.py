"""
Aprendizagem Automatica -- Projeto Final

M12816 - Cristiano Santos
E10973 - Sara Martins
"""

""" BIBLIOTECAS """

import datatest
import nltk
import numpy as np
import pandas as pd
import re
import warnings
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler

""" VARIAVEIS """

colunas = {'id', 'label', 'text', 'label_num'}

""" FUNCOES """

# funcao que mostra caracteristicas relativas ao dataset
def verificar_estatisticas(email):
    # mostrar as primeiras cinco entradas do dataset
    print (email.head())
    # obter uma descricao estatistica do dataset
    print (email.describe())
    # mostrar o par (linhas, colunas) com os valores totais
    print (email.shape)

# funcao que uniformiza as strings de texto de email
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
def preparar_dataframe():
    # abrir o csv com o dataset
    email = pd.read_csv('spam_ham_dataset.csv')

    # verificar a validação do dataset
    datatest.validate(email.columns, colunas)

    # uniformizar a formatacao das strings de texto
    email = reformatar(email)

    return email

# funcao main, que inicializa a execucao do codigo
if __name__ == "__main__":
    email = preparar_dataframe()
    verificar_estatisticas(email)


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