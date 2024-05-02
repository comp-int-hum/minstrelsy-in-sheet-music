from PIL import Image
import gzip
import io
import pytesseract 
import fitz
import argparse 
import zipfile
import json 
import logging


def perform_ocr(pages):
    page_texts = []
    for page in pages:
        pixmap = page.get_pixmap()
        png_bytes = pixmap.tobytes()
        page_image = Image.open(io.BytesIO(png_bytes))
        page_text = pytesseract.image_to_string(page_image)
        page_texts.append(page_text)
    return "\n".join(page_texts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_archive", dest='input_archive', help="zip data for the ocr")
    parser.add_argument("--id_file", dest="id_file", help="file containing list of Levy IDs to process")
    parser.add_argument("--output_file", dest= "output_file", help = "output jsonl file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ids_to_process = set()
    with gzip.open(args.id_file, "rt") as ifd:
        for line in ifd:
            ids_to_process.add(line.strip())
    
    metadata = {}
    with zipfile.ZipFile(args.input_archive, "r") as zip_file, gzip.open(args.output_file, "wt") as ofd:
        with zip_file.open("levy_metadata.jsonl", "r") as ifd:
            for line in io.TextIOWrapper(ifd):
                entry = json.loads(line)
                if entry["levy_pid"] in ids_to_process:
                    metadata[entry["levy_pid"]] = entry    
        for i, name in enumerate(zip_file.namelist()):
            if name.endswith('.pdf'):
                pid = name.replace("-", ":", 1).replace("-", ".").replace(".pdf", "")
                if pid in ids_to_process:
                    with zip_file.open(name) as ifd: 
                        image = fitz.open("pdf", ifd.read())
                        entry = metadata[pid]
                        entry["full_text"] = perform_ocr(image)
                        ofd.write(json.dumps(entry) + "\n")
