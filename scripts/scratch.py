import json

with open("work/random_data_segments_topic_no_5.json", "r") as in_file:
    counter_list = []
    thing = json.load(in_file)
    for x in thing:
        counter_list.append(x)
        print(x)
        print("")
print(len(counter_list))
        
