from optimizer import Optimizer
from preprocess import Preprocessing
from statistics_calculation import Statistics

if __name__ == '__main__':
    data = Preprocessing()
    opt = Optimizer()
    opt.optimize(2015)
    statistics = Statistics(data.data, opt.poly)
    opt.set_polygons(statistics.clusters)
    opt.store_polygons()
