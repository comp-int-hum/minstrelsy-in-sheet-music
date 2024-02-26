import json
import argparse
import numpy 
import pickle
from scipy.spatial.distance import jensenshannon

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    dest="data_file",
    required=True,
    help="data file for analysis")

parser.add_argument(
    "--sort_variable",
    dest="sort_variable",
    required=True,
    help="output file"
)

parser.add_argument(
    "--out_file",
    dest="out_file",
    required=True,
    help="output file"
)
args = parser.parse_args()

jensen_shannon_array = numpy.zeros((23,23), dtype = float)
general_list = []
composer_dictionary = {}

with open (args.data_file, "r") as in_file:
    for x in in_file:
        general_list.append(json.loads(x))

for song in general_list:
    if composer_dictionary.get(song[args.sort_variable]) != None:
          composer_dictionary[song[args.sort_variable]].append(song)
    else:
        composer_dictionary[song[args.sort_variable]] = []
        composer_dictionary[song[args.sort_variable]].append(song)

#for key, value in composer_dictionary.items():
 #   print(key)
  #  print(value)

#for item in composer_dictionary.items():
 #   print(f"Key: '{item[0]}' - Value: {item[1]} (Length: {len(item[1])})")
        
sorted_dict_descending = dict(sorted(composer_dictionary.items(), key=lambda item: len(item[1]), reverse=True))

#for x in sorted_dict_descending:
    #print(x)n    #print(len(sorted_dict_descending[x]))
#organizing ndata based on descending


for entry in sorted_dict_descending["Oliver Ditson, 115 Washington St."]:
    sorted_dict_descending["Oliver Ditson & Co., 277 Washington St"].append(entry)

for entry in sorted_dict_descending["Firth, Pond & Co., 547 Broadway"]:
    sorted_dict_descending["Wm. A. Pond & Co., 547 Broadway"].append(entry)

for entry in sorted_dict_descending["Lee & Walker, 722 Chesnut St."]:
    sorted_dict_descending["Lee & Walker, 722 Chestnut St."].append(entry)

for entry in sorted_dict_descending["Oliver Ditson & Co., 451 Washington St."]:
    sorted_dict_descending["Oliver Ditson & Co., 277 Washington St"].append(entry)    

for entry in sorted_dict_descending["Wm. A. Pond & Co., 25 Union Square, (Broadway, bet. 15th and 16th Sts.)"]:
    sorted_dict_descending["Wm. A. Pond & Co., 547 Broadway"].append(entry)

for entry in sorted_dict_descending["Horace Waters, 333 Broadway"]:
    sorted_dict_descending["Horace Waters, 481 Broadway"].append(entry)
for entry in sorted_dict_descending["Harms, Inc."]:
    sorted_dict_descending["T.B. Harms & Co., 18 East 22nd St."].append(entry)
for entry in sorted_dict_descending["G. Willig"]:
    sorted_dict_descending["Geo. Willig, 171 Chesnut St."].append(entry)    
name_list = ["M. Witmark & Sons", "Oliver Ditson & Co., 277 Washington St.", "Jerome H. Remick & Co.", "Carr\'s Music Store", "Wm. A. Pond & Co., 547 Broadway", "William Hall & Son, 239 Broadway","Leo Feist, Inc.", "John Cole", "Lee & Walker, 722 Chestnut St", "G.E. Blake", "F.D. Benteen", "Ted Snyder Co., 112 West 38th St.", "Chas. K. Harris", "Henry Prentiss, 33 Court St.", "Geo. P. Reed, 17 Tremont Row" ,"S. Brainard\'s Sons", "T.B. Harms", "Atwill, 201 Broadway", "Horace Waters, 481 Broadway" ,"Jos. W. Stern & Co., 102-104 W. 38th St." , "Charles Magnus, 12 Frankfort St.", "G. Willig", "Harms, Inc."]


print(len(name_list))

new_composer_dict = {}
composer_count_list = {}

for x in name_list:
    new_composer_dict[x] = sorted_dict_descending[x]
    print(x)
    print(len(sorted_dict_descending[x]))

for composer, song_list in new_composer_dict.items():

    #print("this is a composer")
    #print(composer)
    groupwise_topic_counts = {}
    for song in song_list:
        for word in song["topics_for_word"]:
            #print(word)
            if word[1] != []: 
                top_topic = word[1][0]
                current_count = groupwise_topic_counts.get(top_topic,0) + 1
                groupwise_topic_counts[top_topic] = current_count

                

    sorted_groupwise_topic_counts = dict(sorted(groupwise_topic_counts.items(), key=lambda item: int(item[0])))
    total = sum(sorted_groupwise_topic_counts.values())
    normalized_topic_counts = {key: value / total for key, value in sorted_groupwise_topic_counts.items()}
    new_normal = {}
    for i in range(20):
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
     #   print("this is p")
      #  print(p)
       # print("this is q")
       # print(q)
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

with open ("work/list_of_top_composers.txt", "w") as out_file:
    json.dumps(output_list)
print(jensen_shannon_array.shape)

numpy.savetxt(args.out_file, jensen_shannon_array, delimiter=',')


