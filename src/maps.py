import asyncio
import folium
import numpy as np
import os
import pandas as pd
import shutil
from src.config import *
from src.frames import convert_html_to_images
import src.helper as helper
from pathlib import Path
from playwright.async_api import async_playwright


def main(n, samples=True, lines=True):

    g, _, lines_df, _ = helper.preprocess_load()

    folium_map = folium.Map(location=CENTER, zoom_start=11, tiles=None)
    folium.TileLayer('OpenStreetMap', opacity=1/5).add_to(folium_map)
    if samples:
        samples_df = pd.read_csv('{0}/results/instances/samples_df_{1}_{2}.csv'.format(RELPATH, FILENAME, n))
        for _, data in g.nodes(data=True):
            data['sample_ct'] = 0
        for _, trip in samples_df.iterrows():
            g.nodes[int(trip.o_node)]['sample_ct'] += 1
            g.nodes[int(trip.d_node)]['sample_ct'] += 1
        for u, data in g.nodes(data=True):
            if data['sample_ct']:
                folium.CircleMarker(
                    location=(data['y'], data['x']), color=HEXBLACK, radius=np.log(1 + data['sample_ct']), weight=0,
                    fill=True, fill_opacity=1, tooltip=u
                ).add_to(folium_map)
    if lines:
        lines_df['width'] = 3/4
        for _, line in lines_df.iterrows():
            folium.PolyLine(
                line.coords, color=line.hexcolor, weight=line.width, opacity=3/4, tooltip=line.name
            ).add_to(folium_map)

    folium_map.save('{0}/results/maps/html/map_{1}_{2}_{3}_{4}.html'.format(
        RELPATH, FILENAME, n, 'samples' if samples else 'nosamples', 'lines' if lines else 'nolines')
    )


if __name__ == '__main__':
    ns = [5, 14, 42, 132, 429, 1430]
    html_dir = './results/maps/html'
    pdf_dir = './results/maps/pdf'
    pdf_crop_dir = './results/maps/pdf_crop'
    png_crop_dir = './results/maps/png_crop'
    cropbox = (475, 175, 825, 675)
    png_zoom = 4
    for n in ns:
        main(n, samples=True, lines=False)
        main(n, samples=False, lines=True)
    asyncio.run(convert_html_to_images(html_dir, pdf_dir, pdf_crop_dir, png_crop_dir, cropbox, png_zoom))

