import argparse
import json
import pickle
import gensim
from gensim.models.ldamodel import LdaModel
import numpy as np
from matplotlib.figure import Figure
from matplotlib.table import Table, table
import matplotlib.pyplot as plt
import csv 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--counts",
    dest="counts",
    required=True,
    help="Counts file generated by apply script."
)
parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="Model file generated by training script."
)
parser.add_argument(
    "--figure",
    dest="figure",
    required=True,
    help="Image file name to save figure to."
)
parser.add_argument(
    "--title",
    dest="title",
    help="Title for top of figure.",
    default="Variation in topics"
)
parser.add_argument(
    "--xlabel",
    dest="xlabel",
    help="Label for the x axis.",
    default="Group"
)
parser.add_argument(
    "--ylabel",
    dest="ylabel",
    help="Label for the y axis.",
    default="Topic proportion"
)
parser.add_argument(
    "--legend",
    dest="legend",
    help="Label for the legend.",
    default="Topics"
)
parser.add_argument(
    "--topic_name",
    dest="topic_names",
    help="Specify human-readable names for topics, e.g. to call the first topic 'Animals', use '--topic_name 1 Animals'.  Can be specified multiple times.",
    nargs=2,
    action="append",
    default=[]
)

parser.add_argument(
    "--out_file",
    dest="out_file",
    help=" out put file"
)

args = parser.parse_args()

# Read in the topic model (really only necessary to know how to map between 
# integers and words: there are other ways we could have done this without
# needing the model, this is just convenient).
with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())

# Read in the counts generated by the previous script that applied the
# topic model to some documents.
with open(args.counts, "rt") as ifd:
    groupwise_counts = json.loads(ifd.read())

# Create and fill up a matrix of counts (the rows are topics, the columns are groups).
#matrix_of_counts = np.zeros(shape=(model.num_topics, len(groupwise_counts))) 
#groups = []
#for group_number, group in enumerate(groupwise_counts):
 #   group_name = group[0]
  #  topic_counts = group[1]
   # groups.append(group_name)
    #for topic, count in topic_counts.items():
     #   topic_number = int(topic)
      #  matrix_of_counts[topic_number, group_number] = count


# Divide each group's topic-count by the total number of counts for the group
# (i.e. normalize the counts to a distribution).
#matrix_of_counts = (matrix_of_counts / matrix_of_counts.sum(0))

#with open (args.out_file, "w") as out_file:
 #    writer = csv.writer(out_file)
  #   for i in range(20):
   #      local_row = matrix_of_counts[i,:].tolist()
    #     print("this is a local row")
     #    print(local_row)
      #   writer.writerow(local_row)
        


# Plot the proportion of each topic's occurrence for each group.
#
# Note: matplotlib can be rather baroque, and there are other
# libraries, like seaborn or plotnine, that build easier-to-use
# abstractions on top of it.
fig = Figure(figsize=(80,80))
ax = fig.add_axes((.06,.62,.9,.33))
ax2 = fig.add_axes(
    (.06,.1,.9,.35),
    label="Prominent words for each topic",
    frameon=False,
    xticks=[],
    yticks=[]
)

topic_name_lookup = {int(k) : v for k, v in args.topic_names}

ax.stackplot(
    list(reversed(groups)),
    np.flip(matrix_of_counts, 0),
    labels=list(reversed([topic_name_lookup.get(i + 1, i + 1) for i in range(model.num_topics)]))
)

ax.set_title(args.title, fontsize=80, fontweight="bold")
ax.set_xlabel(args.xlabel, fontsize=80)
ax.set_ylabel(args.ylabel, fontsize=80)
ax.set_yticks([], [])
ax.legend(
    reverse=True,
    loc='upper left',
    fontsize=60,
    title=args.legend
)

fs = 15
fp = {
    "size" : fs,
    "weight" : "bold"
}
num_words = 10
#tbl = Table(ax2)
#tbl.auto_set_font_size(False)
#width = .1
#height = .05
    
#for topic_number, words in model.show_topics(model.num_topics, 10, formatted=False):
 #   bg = "lightgrey" if topic_number % 2 == 0 else "white"
  #  tbl.add_cell(
   #     2*topic_number,
    #    0,
     #   text=topic_name_lookup.get(topic_number + 1, "#{}".format(topic_number + 1)),
      #  width=0.04,
#        height=height,
#        edgecolor="white",
#        facecolor="white",
#        fontproperties=fp
 #   )

  #  for word_number, word in enumerate(words):
   #     tbl.add_cell(
    #        2*topic_number,
    #        word_number + 1,
    #        text="{}".format(word[0]),
     #       width=width,
      #      height=height,
  #          edgecolor=bg,
  #          facecolor=bg,
  #          fontproperties={"size" : 20, "weight" : "bold"}
   #     )
   #     tbl.add_cell(
    #        2*topic_number + 1,
     #       word_number + 1,
      #      text="{:03.3f}".format(word[1]),
       #     width=width,
        #    height=height,
         #   edgecolor=bg,
      #      facecolor=bg,
      #      fontproperties={"size" : 15}
      #  )
#ax2.add_table(tbl)

fig.savefig(args.figure)
