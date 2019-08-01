import numpy as np
import argparse

from util import Util
from Rbm import RBM
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_hid', type=int, default=64,
                    help='Number of hidden layer units (latent factors)')
parser.add_argument('--user', type=int, default=22,
                    help='user id to recommend books \
                    to (not all ids might be present)')
parser.add_argument('--data_dir', type=str, default='data', required=True,
                    help='path to dataset')
parser.add_argument('--rows', type=int, default=200000,
                    help='number of rows to be used for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--free_energy', type=bool, default=False,
                    help='Export free energy plot')
parser.add_argument('--verbose', type=bool, default=False,
                    help='Display info after each epoch')
args = parser.parse_args()
"""

def main() :

    util = Util()
    #dir = args.data_dir
    dir = "/home/borin/PycharmProjects/RecomendadorFilmes/dataset/"
    #linhas = args.rows
    rating, toWatch, movies = util.readData(dir)
    numVis = len(rating)
    #freeEnergy = args.free_energy
    train = util.preProcess(ratings = rating)
    valid = None
    freeEnergy = True

    #if (freeEnergy == True) :

    train, valid = util.splitData(train)

    h = 64
    #h = args.num_hid
    user = 22
    #user = args.user
    alpha = 0.01
    #alpha = args.alpha
    #w = np.random.normal(loc = 0, scale = 0.01, size = [numVis, h])

    rbm = RBM(alfa = alpha, hidden = h, numVis = numVis)

    # funcionou até aqui

    epocas = 3
    #epocas = args.epochs

    batchSize = 16
    #batchSize = args.batch_size

    v = True
    #v = args.verbose

    reco, prevW, prevVb, prevHb = rbm.train(train = train, valid = valid,
                                            user = user, epocas = epocas,
                                            batchSize = batchSize, freeEnergy = freeEnergy,
                                            verbose = v)

    unWatch, watch = rbm.calculaScores(rate = rating, movies = movies,
                                       toWatch = toWatch, rec = reco, user = user)

    rbm.export(unWatch, watch)

if __name__ == '__main__':
    main()