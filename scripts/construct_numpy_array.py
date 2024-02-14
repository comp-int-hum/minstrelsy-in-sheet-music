import json
import numpy
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    dest="input_file",
    required=True,
    help="json database")

parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="topic model used"
)

parser.add_argument(
    "--number",
    dest="number",
    required=True,
    type = int, 
    help="how many topics"
)


parser.add_argument(
    "--output_file",
    dest="output_file",
    required=True,
    help="where does the file go?"
)

parser.add_argument(
    "--year_bucket",
    dest="year_bucket",
    required=True,
    type = int, 
    help="how are you organizing years?"
)

args = parser.parse_args()

database = {}
year = {}

# Read in the model that was previously trained and serialized.                                                                            
with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())
print(len(model.id2word))
word_len= len(model.id2word)
topics = args.number
buckets = 130 // args.year_bucket
print(buckets)
overall_array= numpy.zeros((topics,word_len,buckets))


with open(args.input_file, "r") as in_file:
    for x in in_file:
       
       composition =  json.loads(x)
       #someday find a way to include the guesses? 
       #print(composition)
       if composition["pub_date"].isdigit():
           start_year = 1800
           if int(composition["pub_date"]) > 1800 and int(composition["pub_date"]) < 1930:
               #fixing the 1925-30 issue)
               pub_date_bucket = (int(composition["pub_date"]) - start_year) // args.year_bucket  
               if int(composition["pub_date"]) >= 1925:
                   pub_date_bucket = 4
               print("this is the date bucket")
               print(pub_date_bucket)
               for word in composition["topics_for_word_phi"]:
                   #print(word)
                   word_num = word[1]
                   #print("this is the word num") 
               #    print(word_num)
                   #print(word[2])
               
                   word[2].sort( key = lambda x : x[1], reverse = True)
                   #print("this is sorted list")
                   #print(word[2])
                   if len(word[2]) > 0: 
                       top_topic = word[2][0][0]
                       #print("this is the top topic")
                #       print(top_topic)
                       print(buckets)
                       print(composition["pub_date"])
                       overall_array[top_topic,word_num, pub_date_bucket] = overall_array[top_topic,word_num, pub_date_bucket] + 1
                       print(overall_array[ top_topic,word_num, pub_date_bucket])
numpy.savetxt("work/test_slice.csv", overall_array[:, :, 1], delimiter=",")
numpy.save(args.output_file, overall_array)
    
                  
                 
