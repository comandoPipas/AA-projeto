"""
Aprendizagem Automatica -- Projeto Final

M12816 - Cristiano Santos
E10973 - Sara Martins
"""

""" BIBLIOTECAS """

import numpy as np
import pandas as pd

""" VARIAVEIS """

""" CODIGO """

# abrir o csv com o dataset
email = pd.read_csv('spam_ham_dataset.csv')

# mostrar as primeiras cinco entradas do dataset
print (email.head())