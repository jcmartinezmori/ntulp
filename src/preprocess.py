import osmnx as ox
import networkx as nx
import pandas as pd
import requests
from src.config import *


def main():

    print('... working on g.')
    g = ox.graph_from_place(PLACE, network_type='drive', retain_all=False, custom_filter=CUSTOM_FILTER)
    ox.save_graphml(g, '{0}/results/preprocess/g_{1}.graphml'.format(RELPATH, FILENAME))
    g = nx.Graph(g)

    print('... working on stops_df.')
    stops_df = ox.features_from_place(PLACE, STOPS_TAGS).droplevel('element_type')
    stops_df = stops_df[stops_df['geometry'].geom_type == 'Point']
    stops_df['node'] = stops_df.apply(lambda stop: ox.nearest_nodes(g, stop.geometry.x, stop.geometry.y), axis=1)
    stops_df.to_file('{0}/results/preprocess/stops_df_{1}.gpkg'.format(RELPATH, FILENAME), driver='GPKG')

    print('... working on lines_df.')
    lines_query = """
    [out:json];
    area["name"="{0}"]["admin_level"="{1}"]->.searchArea;
    (
      relation["type"="route"]["route"="bus"](area.searchArea);
    );
    out body;
    """.format(PLACE, ADMIN_LEVEL)
    lines_response = requests.get('http://overpass-api.de/api/interpreter', params={'data': lines_query}).json()
    lines = []
    for element in lines_response['elements']:
        stops = []
        for member in element['members']:
            if member['type'] == 'node':
                if member['ref'] in stops_df.index:
                    stop = stops_df.loc[member['ref']].node
                    if not stops or stops[-1] != stop:
                        stops.append(stop)
        seen = set()
        stops = tuple(stop for stop in stops if not (stop in seen or seen.add(stop)))
        if len(stops) >= 6:
            length = 0
            route = []
            for s, t in zip(stops[:-1], stops[1:]):
                if s != t:
                    segment_length, segment_route = nx.bidirectional_dijkstra(g, s, t, weight='length')
                    length += segment_length
                    route.extend(segment_route[:-1])
            route.append(stops[-1])
            route = tuple(route)
            coords = tuple((g.nodes[node]['y'], g.nodes[node]['x']) for node in route)
            hexcolor = HEXCOLORS[len(lines) % len(HEXCOLORS)]
            line = [element['id'], element['tags'], stops, length, route, coords, hexcolor]
            lines.append(line)
    lines_df = pd.DataFrame(lines, columns=['id', 'tags', 'stops', 'length', 'route', 'coords', 'hexcolor'])
    lines_df['length'] /= 1000  # kilometers
    lines_df['dist'] = lines_df.apply(
        lambda line: nx.multi_source_dijkstra_path_length(g, line.stops, weight='length', cutoff=LINES_DIST_CTFF),
        axis=1
    )
    lines_df.set_index('id', inplace=True)
    lines_df.drop_duplicates(subset=['length'], inplace=True)  # routes of same length are (very likely) duplicates
    lines_df.to_pickle('{0}/results/preprocess/lines_df_{1}.pkl'.format(RELPATH, FILENAME))

    print('... working on trips_df.')
    trips_df = pd.read_csv('{0}/data/chicago_data.csv'.format(RELPATH))
    trips_df = trips_df[trips_df['Trip Seconds'] >= TRIP_SECS_LB]
    trips_df = trips_df[trips_df['Trip Miles'] >= TRIP_MILES_LB]
    trips_columns = [
        'Pickup Centroid Longitude',
        'Pickup Centroid Latitude',
        'Dropoff Centroid Longitude',
        'Dropoff Centroid Latitude'
    ]
    trips_df = trips_df[trips_columns]
    trips_df.rename(
        columns={
            'Pickup Centroid Longitude': 'o_x',
            'Pickup Centroid Latitude': 'o_y',
            'Dropoff Centroid Longitude': 'd_x',
            'Dropoff Centroid Latitude': 'd_y'
        }, inplace=True
    )
    trips_df = trips_df.dropna()
    trips_df['o_node'] = trips_df.apply(lambda trip: ox.nearest_nodes(g, trip.o_x, trip.o_y), axis=1)
    trips_df['d_node'] = trips_df.apply(lambda trip: ox.nearest_nodes(g, trip.d_x, trip.d_y), axis=1)
    trips_df.to_csv('{0}/results/preprocess/trips_df_{1}.csv'.format(RELPATH, FILENAME), index=False)


if __name__ == '__main__':
    main()
