import logging
import gzip
import math
import json
import argparse
from gensim.models import Word2Vec


logger = logging.getLogger("train_dtm")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Input, already split into sentences")
    parser.add_argument("--output", dest="output", help="Model output")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="How long to train")
    parser.add_argument("--window_size", dest="window_size", type=int, default=5, help="Skip-gram window size")
    parser.add_argument("--embedding_size", dest="embedding_size", type=int, default=300, help="Embedding size")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None, help="Specify a random seed (for repeatability)")
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args.random_seed:
        random.seed(args.random_seed)

    sentences = []
    with gzip.open(args.input, "rt") as ifd:
        for entry in ifd:
            j = json.loads(entry)
            local_sentences = j["full_text"].split()
            sentences.append(local_sentences)
            #print(sentences)
            #sentences. j["full_text"]
    #for x in sentences:
     #   print(x)
      #  for y in x:
       #     print(y)
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.embedding_size,
        window=args.window_size,
        min_count=1,
        workers=4,
        sg=1,
        epochs=args.epochs,
        seed=args.random_seed
    )

    model.save(args.output)
