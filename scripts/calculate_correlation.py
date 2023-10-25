import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import argparse 


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file1",
    dest="data",
    help="Data file model will be applied to."
)


#this one is write for json vs. jsonl                                                                                                                              
parser.add_argument(
    "--input_file2",
    dest="data2",
    help="Data file model will be applied to"
)

parser.add_argument(
    "--output_file",
    dest="output",
    help="Counts output file."
)

args = parser.parse_args()

with open(args.data, "r") as in_file:
    data = json.load(in_file)

# Convert to DataFrame
df = pd.DataFrame([{**{'Year': year}, **values} for year, values in data])

df.set_index('Year', inplace=True)

# Calculate normalized percentages
df_normalized = df.div(df.sum(axis=1), axis=0)

# Compute the difference in normalized percentages between consecutive years
df_diff = df_normalized.diff().dropna()

df_diff.reset_index(inplace=True) 

# Compute the correlations and significance on the differences
#correlations = df_diff.corr()
#correlations = df_diff.corr().sort_index(axis=0).sort_index(axis=1)
#correlations = df_diff.drop(columns=['Year']).corr().sort_index(axis=0).sort_index(axis=1)

correlations = df_diff.drop(columns=['Year']).astype(float).corr()

correlations = correlations.sort_index(axis=0, key=lambda x: x.astype(int)).sort_index(axis=1, key=lambda x: x.astype(int))


features = df_diff.columns.drop('Year')
p_values = df_diff[features].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*correlations.shape)
p_values -= np.eye(*p_values.shape)


# Test significance
#p_values = df_diff.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*correlations.shape)

#p_values -= np.eye(*p_values.shape)

# Print results
print("Correlations of changes in normalized percentages:")
print(correlations)
print("\nSignificance (p-values):")
print(p_values)

# Optionally, filter significant correlations (e.g., p < 0.05)
significance_level = 0.05
significant_corr = correlations[p_values < significance_level]
print("\nSignificant correlations of changes (p < 0.05):")
print(significant_corr)

df_dict = df.to_dict()
df_diff_dict = df_diff.to_dict()
correlations_dict = correlations.to_dict()

# Combine the dictionaries
data = {
    'df': df_dict,
    'df_diff': df_diff_dict,
    'correlations': correlations_dict
}

with open( args.output, "w") as out_file:
    json.dump(data, out_file, indent = 4)
