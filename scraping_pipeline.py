import os
import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import cv2

# ===============================
# CONFIGURATION
# ===============================
BASE_URL = "https://www.bbc.com/"  # CHANGE THIS
PDF_DIR = "scraped_pdfs"
IMG_DIR = "notice_images"
DATASET_DIR = "dataset"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ===============================
# STEP 1: SCRAPE PDF LINKS
# ===============================
def scrape_pdf_links(url):
    print("[1] Scraping notice links...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            if href.startswith("http"):
                pdf_links.append(href)
            else:
                pdf_links.append(url + "/" + href)

    print(f"Found {len(pdf_links)} PDF notices")
    return pdf_links


# ===============================
# STEP 2: DOWNLOAD PDFs
# ===============================
def download_pdfs(pdf_links):
    print("[2] Downloading PDFs...")
    for i, link in enumerate(pdf_links):
        try:
            response = requests.get(link)
            pdf_path = os.path.join(PDF_DIR, f"notice_{i}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_path}")
        except Exception as e:
            print(f"Failed to download {link}: {e}")


# ===============================
# STEP 3: PDF → IMAGE
# ===============================
def convert_pdfs_to_images():
    print("[3] Converting PDFs to images...")
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            pages = convert_from_path(pdf_path, dpi=200)

            for i, page in enumerate(pages):
                img_name = f"{pdf_file[:-4]}_page_{i}.jpg"
                img_path = os.path.join(IMG_DIR, img_name)
                page.save(img_path, "JPEG")
                print(f"Saved image: {img_path}")


# ===============================
# STEP 4: OPTIONAL PAGE SPLIT
# ===============================
def split_page_image(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    top = img[:h//2, :]
    bottom = img[h//2:, :]

    cv2.imwrite(image_path.replace(".jpg", "_top.jpg"), top)
    cv2.imwrite(image_path.replace(".jpg", "_bottom.jpg"), bottom)


# ===============================
# MAIN PIPELINE
# ===============================
def main():
    pdf_links = scrape_pdf_links(BASE_URL)
    download_pdfs(pdf_links)
    convert_pdfs_to_images()

    print("[4] Dataset ready for manual labeling")
    print("Move images into:")
    print("dataset/Critical/")
    print("dataset/Important/")
    print("dataset/Informational/")
    print("dataset/Low/")


if __name__ == "__main__":
    main()
