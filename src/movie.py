import asyncio
from pathlib import Path
import playwright
from playwright.async_api import async_playwright
import fitz
import os


async def convert_html_to_images():

    html_dir = './results/figures/sequence'
    pdf_dir = './results/figures/pdf_frames'
    pdf_crop_dir = './results/figures/pdf_crop_frames'
    png_crop_dir = './results/figures/png_crop_frames'

    Path(pdf_dir).mkdir(exist_ok=True)
    Path(pdf_crop_dir).mkdir(exist_ok=True)
    Path(png_crop_dir).mkdir(exist_ok=True)

    html_files = sorted(Path(html_dir).glob('*.html'), key=lambda f: f.stat().st_ctime)

    async with async_playwright() as p:

        browser = await p.chromium.launch()
        page = await browser.new_page()

        for i, html_file in enumerate(html_files):

            file_url = html_file.resolve().as_uri()
            pdf_output_path = Path(pdf_dir) / f'frame_{i:04d}.pdf'

            await page.goto(file_url)
            await page.set_viewport_size({"width": 1920, "height": 1080})

            await page.wait_for_load_state("networkidle")
            await page.wait_for_function("""
              () => Array.from(document.querySelectorAll('img.leaflet-tile'))
                         .every(img => img.complete)
            """)

            await page.pdf(
                path=pdf_output_path,
                width="1920",
                height="1080",
                print_background=False
            )

        await browser.close()

    crop_box = (475, 175, 825, 675)

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):

            input_pdf_path = os.path.join(pdf_dir, filename)
            output_pdf_path = os.path.join(pdf_crop_dir, filename)
            output_png_path = os.path.join(png_crop_dir, filename.replace('.pdf', '.png'))

            doc = fitz.open(input_pdf_path)
            page = doc[0]
            crop_rect = fitz.Rect(crop_box)
            page.set_cropbox(crop_rect)
            doc.save(output_pdf_path)

            png_zoom = 4
            matrix = fitz.Matrix(png_zoom, png_zoom)
            pix = page.get_pixmap(matrix=matrix)
            pix.save(output_png_path)
            doc.close()


if __name__ == '__main__':
    asyncio.run(convert_html_to_images())
