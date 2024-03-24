import networkx as nx
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

def main(args):
    with open(args.input_path,'r') as f:
        in_dict = json.load(f)
    out_dict = {}
    for k in in_dict:
        out_dict[k] = in_dict[k]['children'] if 'children' in in_dict[k] else None
    g = nx.DiGraph(out_dict)
    nx.draw(g,with_labels=True)
    plt.savefig(f'fig-{Path(args.input_path).stem}.png',format='PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization parameters')
    parser.add_argument('--input_path', type=str, help='input json file')
    args = parser.parse_args()
    main(args)
