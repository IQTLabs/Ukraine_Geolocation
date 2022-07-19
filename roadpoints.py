#!/usr/bin/env python

import argparse
import osmnx as ox
import pandas as pd

def download(location, network_path):
    graph = ox.graph_from_place(location, network_type='all_private')
    ox.save_graphml(graph, filepath=network_path)


def points(network_path, points_path, num):
    print('0')
    graph = ox.load_graphml(network_path)
    print('1')
    graph = ox.project_graph(graph)
    print('2')
    graph = ox.get_undirected(graph)
    print('3')
    points = ox.utils_geo.sample_points(graph, n=num)
    graph = None
    print('4')
    points = points.to_crs('EPSG:4326')
    print('5')
    points = pd.concat([points.y, points.x], axis=1)
    print('6')
    points.to_csv(points_path, sep=',', header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', action='store_true',
                        help='Download street network')
    parser.add_argument('-p', '--points', action='store_true',
                        help='Select random points')
    parser.add_argument('-n', '--num', type=int, default=1000000,
                        help='Number of points to select')
    parser.add_argument('-l', '--location', default='Ukraine',
                        help='Name of location')
    parser.add_argument('-x', '--network', default='./ukraine.gpkg',
                        help='Path to road network file')
    parser.add_argument('-o', '--output', default='./points.csv',
                        help='Path to points file')
    args = parser.parse_args()
    if args.download:
        download(args.location, args.network)
    if args.points:
        points(args.network, args.output, args.num)
