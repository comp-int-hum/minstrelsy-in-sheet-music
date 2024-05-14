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
    "--random_on",
    dest="random_on",
    default = 0, 
    type = int, 
    help="creates a number of random files equal to the input no."
)

parser.add_argument(
    "--seed",
    dest="seed",
    type = int, 
    help="Counts output file."
)

parser.add_argument(
    "--sub_category",
    dest="sub_category",
    nargs = '*',
    help="allows you to pull out a subcategory -- requires two values, the first is the dictionary key, the second is the desired value."
)

parser.add_argument(
    "--random_length",
    dest="random_length",
    type = int, 
    default = 2000,
    help="length of random dictionary"
)

parser.add_argument(
    "--sub_category_search",
    dest="sub_category_search",
    help="allows you to search out a subcategory -- requires two values, the first is the dictionary key, the second is the desired value."
)

parser.add_argument(
    "--sub_category_search_value",
    dest="sub_category_search_value",
    help="allows you to search out a subcategory -- requires two values, the first is the dictionary key, the second is the desired value."
)


parser.add_argument(
    "--reverse_sub_category",
    dest = "minus_sub_cat",
    default = 0,
    type = int, 
    help = "makes subcategory subtract that subcategory from the general population")

parser.add_argument(
    "--output_file",
    dest = "output_file",
    help = "names out output files")


args = parser.parse_args()
sub_cat = args.sub_category
data_dictionary_list = []
output_dictionary_list = []

with open(args.data, "rt") as ifd:
    for line in ifd:
        dictionary = json.loads(line)
        data_dictionary_list.append(dictionary)


if len(sub_cat) > 0:
    print("length greater than zero")
    dict_key = sub_cat[0]
    print(dict_key)
    desired_value = sub_cat[1] + " " + sub_cat[2]
    print(desired_value)



#things are either going to be random or they are going to be sub-category removals 
    
if abs(args.random_on) > 0:  
    if args.seed > 0:
        #allows you to input a seed for replicability
        random.seed(a = args.seed)
    for x in range(args.random_on):
        dictionary_list = random.sample(data_dictionary_list, args.random_length)
        rep_dict_list = []
        for sub_dict in dictionary_list:
            rep_dict = {}
            rep_dict["text"] = sub_dict["text"]
            rep_dict["subjectSearched"] = sub_dict["subjectSearched"]
            rep_dict["time"] = sub_dict["time"]
            #rep_dict["topics_for_word_phi"] = sub_dict["topics_for_word_phi"]
            rep_dict["pid"] = sub_dict["pid"]
            rep_dict["title"] = sub_dict["title"]
            #rep_dict["sub_doc_bow_dict"] = sub_dict["sub_doc_bow_dict"]
            rep_dict_list.append(rep_dict)

        output_dictionary_list.append(rep_dict_list)
        print("appended dictionary")
elif len(sub_cat) > 0:
        #if you only want things with the subcateogry 
        if args.minus_sub_cat == 0: 
            print("sub_cat true")
            new_list = []
            for row in data_dictionary_list: 
                if row[dict_key] == desired_value:
                    new_list.append(row)
            #        print(new_list)        
            data_dictionary_list = new_list
            output_dictionary_list.append(data_dictionary_list)
        else:
            #the reverse 
            print("reverse_sub_cat_true")
            new_list = []
            for row in data_dictionary_list:
                if row[dict_key] != desired_value:
                    new_list.append(row)
                #    print(new_list)
            data_dictionary_list = new_list
            output_dictionary_list.append(data_dictionary_list)

#allows you to search a sub_category for specific values. 
            
elif args.sub_category_search != "None":
    values = args.sub_category_search_value
    for entry in data_dictionary_list:
        #print(args.sub_category_search)
        #print(args.sub_category_search_value)
        #print(entry[args.sub_category_search])
        #if args.sub_category_search_value in entry[args.sub_category_search]:
         #   output_dictionary_list.append(entry)
        #composers = ["J.W. Johnson", "Bob Cole", "Ernest Hogan", "Will Marion Cook", "Gussie L. Davis", "Williams and Walker", "Chris Smith"]

        if any(element in entry[args.sub_category_search] for element in values):
            print(entry)
            output_dictionary_list.append(entry)    
                
print(len(output_dictionary_list))
with open(args.output_file, "w") as ofd:
        json.dump(output_dictionary_list, ofd)

