import json
import argparse
import random
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be applied to."
)

parser.add_argument(
    "--group_resolution",
    dest="group_resolution",
    default=10,
    type=int,
    help="The size of each group (e.g. number of years, or whatever units 'group_field' uses)."
)

parser.add_argument(
    "--date_cutoff",
    dest="cutoff_date",
    default=1930,
    type=int,
    help="final date counted for groups"
)

parser.add_argument(
    "--date_start",
    dest="start_date",
    default=1800,
    type=int,
    help="final date counted for groups"
)


parser.add_argument(
    "--counts",
    dest="counts",
    required=True,
    help="Counts output file."
)

parser.add_argument(
    "--lost_words",
    dest="lost_words",
    required=True,
    help="Counts skipped words"
)

parser.add_argument(
    "--lost_words_check",
    dest = "lost_words_check",
    default = False,
    type = bool,
    help = "switch for lost words check")



counter = 0
pub_date_lost = 0
lost_words = []
groupwise_topic_counts = {}
counts_list = []


args = parser.parse_args()



with open(args.data, "rt") as ifd:
    for line in ifd:
        dictionary = json.loads(line)
        data_dictionary_list.append(dictionary)

for individual_dictionary in data_dictionary_list: 
        for row in individual_dictionary:
            if row["pub_date"].isdigit():
                group_value = int(row["pub_date"])
                group = group_value - (group_value % args.group_resolution)
                groupwise_topic_counts[group] = groupwise_topic_counts.get(group, {})
                document_topics_by_word = row["topics_for_word_phi"]
                subdocument_bow_lookup = row["sub_doc_bow_dict"]
                #print(subdocument_bow_lookup)
                for word in document_topics_by_word:      
                    word_id = word[1]
                     
                    topics = word[2]
                  
                    if len(topics) > 0:
                        topic = max(topics, key = lambda x: x[1])
                        topic = topic[0]
                    if group_value  < args.cutoff_date and group_value > args.start_date:
                        groupwise_topic_counts[group][topic] = groupwise_topic_counts[group].get(topic, 0) + subdocument_bow_lookup[str(word_id)]
                    else:
                        counter = counter + 1
                        lost_words.append(word)
        counts_list.append(groupwise_topic_counts)                
        counter = str(counter)
        #lost words count might not work anymore, coming back to this later. 
        if args.lost_words_check == True: 
            with open(args.lost_words, "wt") as rt:
                rt.write("this is the lost words counter")
                rt.write(counter)
                rt.write("this is the number of documents without dates")
                rt.write(str(pub_date_lost))
                for word in lost_words:
                    rt.write(str(word))

with open(args.counts, "wt") as ofd:
    for count in counts_list: 
        ofd.write(
            json.dumps(
                [(k, v) for k, v in sorted(groupwise_topic_counts.items()) if len(v) > 0],
                indent=4
            )
        )

