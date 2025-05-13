import asyncio
import fitz
import os
import playwright
import shutil
import subprocess
from pathlib import Path
from playwright.async_api import async_playwright

html_dir = './results/figures/sequence'
pdf_dir = './results/figures/pdf_frames'
pdf_crop_dir = './results/figures/pdf_crop_frames'
png_crop_dir = './results/figures/png_crop_frames'
cropbox = (475, 175, 825, 675)
png_zoom = 4


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
    asyncio.run(convert_html_to_images())
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', '3',
        '-i', 'png_crop_dir/frame_%04d.png',
        '-crf', '18',
        '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        'sequence.mp4'
    ]
    subprocess.run(ffmpeg_command)

