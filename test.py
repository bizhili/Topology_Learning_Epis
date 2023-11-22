import argparse

parser = argparse.ArgumentParser(description='Topology fitting parameters')

parser.add_argument('--randomGraph', type=str, default="GEO", help='Choosing random graph model: GEO(defult), ER, WS, BA')

args = parser.parse_args()
print(args.l)