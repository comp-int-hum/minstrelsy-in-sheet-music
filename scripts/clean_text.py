import logging
import gzip
import json
import argparse
import re


logger = logging.getLogger("clean_text")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--min_length", dest="min_length", default=3, type=int)
    parser.add_argument("--min_year", dest="min_year", type=int)
    parser.add_argument("--max_year", dest="max_year", type=int)
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            year = re.sub(r"\D+", "", j["pub_date"])
            if year.isdigit() == True:
                year = int(year)
                if (args.min_year and year < args.min_year) or (args.max_year and year > args.max_year):
                    continue
            else:
                continue
            j["pub_date"] = str(year)
            tokens = [
                re.sub(r"[^a-zA-Z0-9]", "", w.lower()) for w in re.sub(r"\s*\-\s*", "", j["full_text"]).split()
            ]            
            j["full_text"] = " ".join(
                [t for t in tokens if len(t) >= args.min_length]
            )
            ofd.write(json.dumps(j) + "\n")
