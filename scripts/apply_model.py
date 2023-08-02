from itertools import islice
import csv
import re
import logging
import random
import pickle
import json
import argparse
import gensim 
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim.parsing.preprocessing as gpp


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="Model file generated by training script.")

parser.add_argument(
    "--topic_number",
    dest="topic",
    required=True,
    help="number of topics"
)


parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be applied to."
)

parser.add_argument(
    "--counts",
    dest="counts",
    required=True,
    help="Counts output file."
)

parser.add_argument(
    "--group_resolution",
    dest="group_resolution",    
    default=10,
    type=int,
    help="The size of each group (e.g. number of years, or whatever units 'group_field' uses)."
)

parser.add_argument(
    "--minimum_word_length",
    dest="minimum_word_length",
    default=3,
    type=int,
    help="Minimum word length"
)
args = parser.parse_args()

# A silly default setting in the csv library needs to be changed 
# to handle larger fields.
csv.field_size_limit(1000000000)
no_topics = args.topic
# For each "group", we'll collect the number of times each topic
# occurs.
groupwise_topic_counts = {}
data_dictionary_list = []
group_names = {}
# Read in the model that was previously trained and serialized.
with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())

# Open your file to read ("r") in text mode ("t") as a variable 
# ("ifd" is what Tom uses when reading files, it stands for 
# "input file descriptor").
with open(args.data, "rt") as ifd:
    dicti = gensim.corpora.dictionary.Dictionary.load("work/dictionary_{}_topics.gensim".format(no_topics))
    print(len(dicti))
    # Use the file handle to create a CSV file handle, specifying 
    # that the delimiter is actually <TAB> rather than <COMMA>.
    for line in ifd:
        dictionary = json.loads(line)
        data_dictionary_list.append(dictionary)

    # Iterate over each row of your file: since we used DictReader 
    # above, each row will be a dictionary.
    for row in data_dictionary_list:
        
        if row["pub_date"].isdigit():
            group_value = int(row["pub_date"])
           # print("did it!")
            group = group_value - (group_value % args.group_resolution)
        # Make sure there is a bucket for the group.
            groupwise_topic_counts[group] = groupwise_topic_counts.get(group, {})
        
        # We want to prepare the data the same way we prepared the data that
        # trained the model (there may be situations where we'd do something
        # different, but only with a particularly good reason!).
            tokens = gpp.split_on_space(
            gpp.strip_short(
                gpp.remove_stopwords(
                    gpp.strip_non_alphanum(
                        row["full_text"].lower()
                    ),
                ),
                minsize=args.minimum_word_length
            )
           )
            
          #  print(tokens)
            
            #ls = []
            #for y in tokens:
                #y = y.lower()
                #ls.append(y)
            # Turn the subdocument tokens into integers and count them, using the
            # trained model (so it employs the same mapping as it was trained with).
    
            subdocument_bow = dicti.doc2bow(tokens)
           # print("trigger")
           # print(row["full_text"])
           # print(subdocument_bow)
           # print(len(subdocument_bow))
           #subdocument_bow = model.id2word.doc2bow(tokens)
           
            # It will be useful to have the "bag-of-words" counts as a dictionary, too.
            subdocument_bow_lookup = dict(subdocument_bow)
           # print(subdocument_bow)
            # Apply the model to the subdocument, asking it to give the specific
            # assignments, i.e. which topic is responsible for each unique word.
            # Note how an underscore ("_") can be used, here and in other situations,
            # when you *don't* want to assign something to a variable, because you
            # aren't going to need it.  This makes the code (and your intentions)
            # clear.
            A, labeled_subdocument, B = model.get_document_topics(
                subdocument_bow,
                per_word_topics=True
            )

            #for x in islice(A,5):
             #   print("this is A")
              #  print(x)
            #for x in islice(labeled_subdocument,5):
             #   print("this is labeled_subdocument")
              #  print(x)
            #for x in islice(B,5):
             #   print("this is b")
              #  print(x)
        
            #labeled_list.append(labeled_subdocument)
            
            # Add the topic counts for this subdocument to the appropriate group.
            for word_id, topics in labeled_subdocument:
                # Gensim insists on returning *lists* of topics in descending order of likelihood.  When
                # the list is empty, it means this word wasn't seen during training (I think!), so we skip
                # it.
                if len(topics) > 0:
                    
                    # Assume the likeliest topic.
                    topic = topics[0]
                    
                    # Add the number of times this word appeared in the subdocument to the topic's count for the group.
                    groupwise_topic_counts[group][topic] = groupwise_topic_counts[group].get(topic, 0) + subdocument_bow_lookup[word_id]
                    #groupwise_topic_counts[group][topic] = groupwise_topic_counts[group].get(topic,0) + 1  
#the count should probably be altered to have more information 





# Save the counts to a file in the "JSON" format.  The 'indent=4' argument makes it a lot easier
# for a human to read the resulting file directly.
with open(args.counts, "wt") as ofd:
    ofd.write(
        json.dumps(
            [(k, v) for k, v in sorted(groupwise_topic_counts.items()) if len(v) > 0],
            indent=4
        )
    )