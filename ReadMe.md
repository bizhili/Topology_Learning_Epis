usage: run.py [-h] [--randomGraph RANDOMGRAPH] [--seed SEED] [--n N] [--strains STRAINS] [--epoches EPOCHES]
              [--device DEVICE] [--weightModel WEIGHTMODEL] [--intense INTENSE] [--R0Mean R0MEAN] [--R0Std R0STD]
              [--tauMean TAUMEAN] [--tauStd TAUSTD] [--modelLoad MODELLOAD] [--dense DENSE]

Topology fitting parameters

optional arguments:
  -h, --help            show this help message and exit
  --randomGraph RANDOMGRAPH
                        Choosing random graph model(str): GEO(defult), ER, WS, BA
  --seed SEED           Setting random seed(int): 10(defult)
  --n N                 Setting nodes number(int): 50(defult)
  --strains STRAINS     Setting strains number(int): 1(defult)~4
  --epoches EPOCHES     Setting stop epoches number(int): 100000(defult)
  --device DEVICE       Setting device(str): cuda:0(defult), cpu
  --weightModel WEIGHTMODEL
                        Setting adjacency weight model(str): degree(defult), gravity, identical
  --intense INTENSE     Setting the intense of selecting nodes degree from low to high(int): 0(defult), 1, 2
  --R0Mean R0MEAN       Setting the mean value of R0s, average distribution (float): 8.3(defult)
  --R0Std R0STD         Setting the Std value of R0s, average distribution (float): 4(defult)
  --tauMean TAUMEAN     Setting the mean value of R0s, average distribution (float): 6.2(defult)
  --tauStd TAUSTD       Setting the Std value of R0s, average distribution (float): 0.1(defult)
  --modelLoad MODELLOAD
                        Setting load model (string): AA(defult), AB, BA, BB, infer2018
  --dense DENSE         Setting avg degree of BA, WS, ER (int): 8(defult)