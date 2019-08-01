import pandas as pd
import numpy as np
import random as r
import os

class Util(object) :

    def readData(self, folder) :
        """
        Função que lê os dados para
        construção do modelo de recomendação

        :param folder:
        :return:
        """

        print("lendo o database")
        ratings = pd.read_csv(os.path.join(folder, "testeRating.csv"))
        toWatch = pd.read_csv(os.path.join(folder, "testeToWatch.csv"))
        movies = pd.read_csv(os.path.join(folder, "testeMovies.csv"))

        return ratings, toWatch, movies

    def limpaSubset(self, ratings, numLin) :
        """
        Função que limpa o database

        :param df:
        :param numLin:
        :return:
        """

        print("extraindo as colunas do database")
        temp = ratings.sort_values(by = ["userID"], ascending = True)
        ratings = temp.iloc[:numLin, :]

        return ratings

    def preProcess(self, ratings) :
        """
        Função que faz o pré-processamento
        para alimentar a rede

        :param df:
        :return:
        """

        print("pré-processando o dataset")
        ratings = ratings.reset_index(drop = True)
        ratings["List Index"] = ratings.index
        usersGroup = ratings.groupby("UserId")

        total = []

        for userId, curUser in usersGroup :

            temp = np.zeros(len(ratings))

            for num, movie in curUser.iterrows() :

                val = int(movie["List Index"])
                temp[val] = movie["Rating"]

            total.append(temp)

        return total

    def splitData(self, totalData) :
        """
        Função para separar os dados em treinamento e validação

        :param totalData:
        :return:
        """

        print("Liberando a energia necessária, dividindo em treinamento e validação")
        r.shuffle(totalData)
        n = len(totalData)

        print("O tamanho total dos dados é : {}".format(n))
        tamTrain = int(n * 0.35)

        XTrain = totalData[:tamTrain]
        YTrain = totalData[tamTrain:]

        print("Tamanho do dataset de treinamento : {}".format(len(XTrain)))
        print("Tamanho do dataset de validação : {}".format(len(YTrain)))

        return XTrain, YTrain

    def freeEnergy(self, vSample, W, vB, hB) :
        """
        Função que calcula a energia livre

        :param vSample:
        :param W:
        :param vB:
        :param hB:
        :return:
        """

        wxB = np.dot(vSample, W) + hB
        vBiasTerm = np.dot(vSample, vB)
        hiddenTerm = np.sum(np.log(1 + np.exp(wxB)), axis = 1)

        return hiddenTerm - vBiasTerm