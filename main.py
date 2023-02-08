from optimizer import Optimizer
from preprocess import Preprocessing

if __name__ == '__main__':
    Preprocessing()
    opt = Optimizer()
    opt.optimize(2015)

