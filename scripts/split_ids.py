import gzip
import io
import argparse 
import zipfile
import json 
import math


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_archive", dest='input_archive', help="Zip file")
    parser.add_argument(dest= "output_files", nargs="+", help="Any number of output files to evenly split the IDs amongst")
    args = parser.parse_args()

    all_ids = []
    with zipfile.ZipFile(args.input_archive, "r") as zip_file:
        with zip_file.open("levy_metadata.jsonl", "r") as ifd:
            for line in io.TextIOWrapper(ifd):
                entry = json.loads(line)
                all_ids.append(entry["levy_pid"])

    per_file = math.ceil(len(all_ids) / len(args.output_files))
    for i, file_name in enumerate(args.output_files):
        with gzip.open(file_name, "wt") as ofd:
            ofd.write("\n".join(all_ids[per_file * i : per_file * (i + 1)]))
