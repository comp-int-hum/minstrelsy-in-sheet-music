import gzip
import argparse 
import json 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", dest='output', help="Merged output file")
    parser.add_argument(dest="input_files", nargs="+", help="Any number of input files to concatenate")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd:
        for fname in args.input_files:
            with gzip.open(fname, "rt") as ifd:
                for line in ifd:
                    ofd.write(line)
