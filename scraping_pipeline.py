import os
import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path

BASE_URLS = [
    "https://www.bbc.com/news",
    "https://indianexpress.com/section/india/"
]

PDF_DIR = "scraped_pdfs"
IMG_DIR = "scraped_images"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def scrape_and_download():
    for base_url in BASE_URLS:
        print(f"Scraping: {base_url}")
        try:
            r = requests.get(base_url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.endswith(".pdf"):
                    if not href.startswith("http"):
                        href = base_url + href
                    pdf_name = href.split("/")[-1]
                    pdf_path = os.path.join(PDF_DIR, pdf_name)

                    pdf_data = requests.get(href)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_data.content)
                    print(f"Downloaded {pdf_name}")
        except Exception as e:
            print("Error:", e)

def pdf_to_images():
    for pdf in os.listdir(PDF_DIR):
        if pdf.endswith(".pdf"):
            pages = convert_from_path(os.path.join(PDF_DIR, pdf), dpi=200)
            for i, page in enumerate(pages):
                img_path = os.path.join(
                    IMG_DIR, f"{pdf[:-4]}_page_{i}.jpg"
                )
                page.save(img_path, "JPEG")
                print("Saved", img_path)

if __name__ == "__main__":
    scrape_and_download()
    pdf_to_images()
