import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--counts",
    dest="data",
    required=True,
    help="counts file program will be applied to."
)

parser.add_argument(
    "--output",
    dest="output",
    required=True,
    help="output json file ."
)

args = parser.parse_args()

def compute_variances_single_line(input_file):
    # Load the data
    with open(input_file, 'r') as f:
        data_list = []
        for line in f:
            data_list.append(json.loads(f)
        results_list = []
        for entry in data_list: 
            data = {item[0]: item[1] for item in entry}

            # Identify all unique topics
            all_topics = set()
            for year in data:
                all_topics.update(data[year].keys())
                all_topics = list(all_topics)

                # Compute the average proportion of each topic across the entire dataset
                total_word_counts = sum([sum(year_data.values()) for year_data in data.values()])
                avg_proportions = {topic: sum([year_data.get(topic, 0) for year_data in data.values()]) / total_word_counts for topic in all_topics}

                # Compute the overall variance for each topic across the dataset
                overall_variance_by_topic = {}
            for topic in all_topics:
                proportions = [data[year].get(topic, 0) / sum(data[year].values()) for year in data]
                overall_variance_by_topic[topic] = np.var(proportions)

                # Compute the overall variance of the dataset
                all_proportions = []
            for year in data:
                year_proportions = [data[year].get(topic, 0) / sum(data[year].values()) for topic in all_topics]
                all_proportions.append(year_proportions)
                overall_variance = np.mean(np.var(all_proportions, axis=0))

                # Organize results into desired format
                results = {
                    # "variance_by_year": variance_by_year,
                    "overall_variance_by_topic": overall_variance_by_topic,
                    "overall_variance": overall_variance
                }
            results_list.append(results)
    return results_list
        

results_list = compute_variances_and_organize(args.data)

# If you want to save the results to a JSON file:

with open(args.output, 'w') as f:
    json.dump(results, f,  indent=4)

