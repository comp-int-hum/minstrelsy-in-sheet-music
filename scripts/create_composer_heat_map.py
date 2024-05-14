import json
import argparse
import numpy 
import pickle
from scipy.spatial.distance import jensenshannon

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    dest="data_file",
    required=True,
    help="data file for analysis")

parser.add_argument(
    "--topic_num",
    dest="topic_num",
    required=True,
    type = int, 
    help="number of topics")


parser.add_argument(
    "--output_file",
    dest="out_file",
    required=True,
    help="output file"
)
args = parser.parse_args()

#jensen_shannon_array = numpy.zeros((23,23), dtype = float)
general_list = []
composer_dictionary = {}

with open (args.data_file, "r") as in_file:
    general_list =json.load(in_file)


for song in general_list:
    print(song)
    print("")

        
for song in general_list:
    print(song)
    print(song["dc.contributor.other"])
    if composer_dictionary.get(song["dc.contributor.other"], "fail") == "fail":
       composer_dictionary[song["dc.contributor.other"]] = [song] 
    else:
        composer_dictionary[song["dc.contributor.other"]].append(song)

sorted_dict_descending = dict(sorted(composer_dictionary.items(), key=lambda item: len(item[1]), reverse=True))

new_composer_dict = {}
composer_count_list = {}

length = len(sorted_dict_descending)
jensen_shannon_array = numpy.zeros((length, length), dtype = float)



for composer, song_list in sorted_dict_descending.items():

    groupwise_topic_counts = {key: 0 for key in range(args.topic_num)} 
    for song in song_list:
        for word in song["text"]:
            #print(word)
            
                
            current_count = groupwise_topic_counts.get(word[1],0) + 1
            groupwise_topic_counts[word[1]] = current_count

    sorted_groupwise_topic_counts = dict(sorted(groupwise_topic_counts.items(), key=lambda item: int(item[0])))
    total = sum(sorted_groupwise_topic_counts.values())
    normalized_topic_counts = {key: value / total for key, value in sorted_groupwise_topic_counts.items()}
    new_normal = {}
    
    for i in range(len(groupwise_topic_counts)):
        new_normal[i] = normalized_topic_counts.get(i,0)
    composer_count_list[composer] = new_normal
row_counter = 0

for composer, counts in composer_count_list.items():
    distribution_1 = counts
    column_counter = 0
    for second_composer, second_counts in composer_count_list.items():
        distribution_2 = second_counts
        p = numpy.array([distribution_1.get(count,0) for count in counts])
        q = numpy.array([distribution_2.get(a_count,0) for a_count in second_counts])
        print("this is p")
        print(p)
        print("this is q")
        print(q)
        divergence = jensenshannon(p,q)
        print(divergence)
        print([row_counter, column_counter])
        jensen_shannon_array[row_counter,column_counter] = divergence
        column_counter = column_counter + 1
    row_counter = row_counter + 1
        #code this code includes the date value -- skipping it for now 
        # if song["pub_date"].isdigit():
        #        group_value = int(song["pub_date"])
         #       group = group_value - (group_value % args.group_resolution)
          #      groupwise_topic_counts[group] = groupwise_topic_counts.get(group, {})
           #     for word in song["document_topics_word"]:
            #        top_topic = word[0][1][0]
             #       current_count = groupwise_topic_counts[group].get(top_topic,0) + 1
              #      groupwise_topic_counts[group]["top_topic"] = current_count
output_list = []

for x in composer_count_list:
    output_list.append(x)

#with open ("work/list_of_top_composers.txt", "w") as out_file:
#    json.dumps(output_list)
#print(jensen_shannon_array.shape)

numpy.savetxt(args.out_file, jensen_shannon_array, delimiter=',')


