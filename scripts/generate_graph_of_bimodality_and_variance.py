import numpy
import argparse
import json
import pickle
import gensim
import gensim
import csv 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--matrix",
    dest="matrix",
    required=True,
    help="numpy matrix")

parser.add_argument(
    "--output_dictionary",
    dest="output_file",
    help="where does the file go?"
)

parser.add_argument(
    "--output_chart",
    dest="output_chart",
    help="where does the file go?"
)


parser.add_argument(
    "--model",
    dest="model",
    help="model used for creating dataset"
)

args = parser.parse_args()

matrix = numpy.load(args.matrix)




# creating the variance measure -- get the sum of each word distribution at each time slot index 3) 

sum_over_topics = numpy.sum(matrix, axis=0, keepdims=True)

#making sure you never divide by zero
epsilon = 1e-8

smoothed_sum_over_topics = sum_over_topics + epsilon

#dividing each part of the matrix by the sum to get normalized  percentages 

normalized_matrix_2 = numpy.divide(matrix, smoothed_sum_over_topics)


#mean_over_time_normalized = numpy.mean(normalized_matrix_2, axis=2)

#getting normalized variance of each word/topic combo ove time 

variance_over_time_normalized = numpy.var(normalized_matrix_2, axis=2)


#then getting the average variance of the whole 

average_variance_per_word = numpy.mean(variance_over_time_normalized, axis=0)



total_average_variance_per_word = numpy.mean(average_variance_per_word)


summed_matrix = numpy.sum(matrix, axis =2)

column_sums = numpy.sum(summed_matrix, axis = 0)
#print(column_sums)
normalized_matrix = summed_matrix / column_sums



qualifying_columns = []

for col_index in range(normalized_matrix.shape[1]):
    
    sorted_column = numpy.sort(normalized_matrix[:, col_index])[::-1]
    
    # Check the difference between the top two values
    
    if numpy.sum(summed_matrix[:, col_index]) > 100: 
            #if normalized_matrix[18,col_index] > .2: 
        number = sorted_column[0] - sorted_column[1]
        #print("this is number")
        #print(number)
        value = [col_index,number] 
        qualifying_columns.append(value)


with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())
        
data_list = []
#print(qualifying_columns)            
for entry in qualifying_columns:
    #print(entry)
    variance = average_variance_per_word[entry[0]]
    entry.append(variance)
    data_list.append(entry)
    

#print(data_list)

word_ids = [item[0] for item in data_list]
bi_modalities = [item[1] for item in data_list]
average_variances = [item[2] for item in data_list]
    

plt.figure(figsize=(12, 8))  # Adjust figure size as needed

# Step 2: Create the scatter plot and annotate points
for word_id, bi_modality, average_word_variance in data_list:
    plt.scatter(bi_modality, average_word_variance, alpha=0.5)
    #plt.annotate(word_id, (bi_modality, average_word_variance), fontsize=9, alpha=0.75)
    
# Customize the plot
plt.title('Bimodality vs Variance of Words')
plt.xlabel('Bimodality Score')
plt.ylabel('Average Variance')
plt.grid(True)

# Set the limits of the axes
plt.xlim(0, 1)
plt.ylim(0, .02)

plt.savefig(args.output_chart, dpi=300, bbox_inches='tight')


# Convert data_list to numpy array for vectorized operations
data_array = numpy.array(data_list, dtype=object)
bi_modalities = data_array[:, 1].astype(float)
average_variances = data_array[:, 2].astype(float)

# Define corners
corners = {
    'Bottom-Left': (0, 0),
    'Top-Left': (0, 1),
    'Top-Right': (1, 1),
    'Bottom-Right': (1, 0),
}

closest_words = {}  # Store the 10 closest words to each corner
for corner_name, (cx, cy) in corners.items():
    # Calculate Euclidean distance to the corner for each word
    distances = numpy.sqrt((bi_modalities - cx) ** 2 + (average_variances - cy) ** 2)
    closest_indices = numpy.argsort(distances)[:10]
    closest_words[corner_name] = data_array[closest_indices].tolist()


output_dictionary = {}
    
for corner, words in closest_words.items():
    for word in words:
        real_word = model.id2word[word[0]]
        word.append(real_word)
        

    
print(closest_words)

    
#bi_modal_word_indice_list = find_close_top_values_columns(normalized_matrix, threshold = 0.2)
#bi_modal_well_attested_indice_list = []
#for value in bi_modal_word_indice_list:
 #   if column_sums[value] >= 50:
  #      bi_modal_well_attested_indice_list.append(value)

#hi_variance_well_attested_bi_modal = []
        
#for value in bi_modal_well_attested_indice_list:
 #   if average_variance_per_word[value] > total_average_variance_per_word:
  #      hi_variance_well_attested_bi_modal.append(value)

#print(bi_modal_word_indice_list)


#bi_modal_word_list = []

#for word_id in  hi_variance_well_attested_bi_modal:
 #   bi_modal_word_list.append([word_id, model.id2word[word_id], column_sums[word_id]] + list(normalized_matrix[:,word_id]))



with open(args.output_file, "w") as out_file:
    json.dump(closest_words, out_file)

#print(len(bi_modal_word_list))


#for entry in bi_modal_word_list:
 #   print(entry)
  #  print(summed_matrix[:,entry[0]])
