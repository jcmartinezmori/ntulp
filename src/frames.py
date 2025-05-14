import asyncio
import fitz
import folium
import numpy as np
import os
import pandas as pd
import playwright
import pickle
import shutil
import src.helper as helper
from src.config import *
import subprocess
from pathlib import Path
from playwright.async_api import async_playwright

objective = 'maximin'
blocking_TimeLimit = 600 * 1
blocking_EpsLimit = 0
modelname = 'v2-{0}-{1}-{2}'.format(objective, blocking_TimeLimit, blocking_EpsLimit)
blocking_IterCountStart = 0
blocking_IterCountEnd = 100

html_dir = './results/frames/html'
pdf_dir = './results/frames/pdf'
pdf_crop_dir = './results/frames/pdf_crop'
png_crop_dir = './results/frames/png_crop'
cropbox = (475, 175, 825, 675)
png_zoom = 4


def plot_frames():

    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    os.makedirs(html_dir)

    g, lines_df = helper.load()
    samples_df = pd.read_csv('{0}/results/instances/samples_df_{1}.csv'.format(RELPATH, FILENAME))

    for blocking_IterCount in range(blocking_IterCountStart, blocking_IterCountEnd + 1):

        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'rb') as file:
            x_N, _, _, _, _, S, _ = pickle.load(file)

        lines_df['width'] = [LINESCALING * x_N[j] / lines_df.iloc[j].length for j in range(len(x_N))]
        for _, data in g.nodes(data=True):
            data['sample_ct'] = 0
        for i, sample in samples_df.iterrows():
            if i in S:
                g.nodes[sample.o_node]['sample_ct'] += 1
                g.nodes[sample.d_node]['sample_ct'] += 1

        folium_map = folium.Map(location=CENTER, zoom_start=11, tiles=None)
        # folium.TileLayer('OpenStreetMap', opacity=1/5).add_to(folium_map)
        for _, line in lines_df.iterrows():
            folium.PolyLine(
                line.coords, color=line.hexcolor, weight=line.width, opacity=1, tooltip=line.name
            ).add_to(folium_map)
        folium_map.save('{0}/pre1_{1}_{2}_{3}.html'.format(html_dir, FILENAME, modelname, blocking_IterCount))
        for u, data in g.nodes(data=True):
            if data['sample_ct']:
                folium.CircleMarker(
                    location=(data['y'], data['x']), color=HEXBLACK, radius=np.log(1 + 1 * data['sample_ct']), weight=0,
                    fill=True, fill_opacity=1, tooltip=u
                ).add_to(folium_map)
        folium_map.save('{0}/pos1_{1}_{2}_{3}.html'.format(html_dir, FILENAME, modelname, blocking_IterCount))
        folium_map.save('{0}/pos2_{1}_{2}_{3}.html'.format(html_dir, FILENAME, modelname, blocking_IterCount))


async def convert_html_to_images():

    if os.path.exists(pdf_dir):
        shutil.rmtree(pdf_dir)
    os.makedirs(pdf_dir)
    if os.path.exists(pdf_crop_dir):
        shutil.rmtree(pdf_crop_dir)
    os.makedirs(pdf_crop_dir)
    if os.path.exists(png_crop_dir):
        shutil.rmtree(png_crop_dir)
    os.makedirs(png_crop_dir)

    html_files = sorted(Path(html_dir).glob('*.html'), key=lambda f: f.stat().st_ctime)

    async with async_playwright() as p:

        browser = await p.chromium.launch()
        page = await browser.new_page()

        for i, html_file in enumerate(html_files):

            file_url = html_file.resolve().as_uri()
            pdf_out = Path(pdf_dir)/f'frame_{i:04d}.pdf'

            await page.goto(file_url)
            await page.set_viewport_size({'width': 1920, 'height': 1080})
            await page.wait_for_load_state('networkidle')
            await page.pdf(path=pdf_out, width='1920', height='1080', print_background=False)

        await browser.close()

    pdf_files = sorted(Path(pdf_dir).glob('*.pdf'), key=lambda f: f.stat().st_ctime)
    for pdf_file in pdf_files:

        pdf_crop_out = Path(pdf_crop_dir)/pdf_file.name
        png_crop_out = Path(png_crop_dir)/pdf_file.name.replace('.pdf', '.png')

        doc = fitz.open(pdf_file)
        page = doc[0]
        page.set_cropbox(fitz.Rect(cropbox))
        doc.save(pdf_crop_out)

        pix = page.get_pixmap(matrix=fitz.Matrix(png_zoom, png_zoom))
        pix.save(png_crop_out)
        doc.close()


if __name__ == '__main__':
    try:
        plot_frames()
    except FileNotFoundError:
        pass
    asyncio.run(convert_html_to_images())
    """
    ffmpeg -framerate 3 -i results/frames/png_crop/frame_%04d.png -crf 18 -preset slow -pix_fmt yuv420p results/frames/frames.mp4
    """
