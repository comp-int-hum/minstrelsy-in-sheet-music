import json


with open("work/percentage_report_findings.json", "r") as read_file:
    doc_list = json.load(read_file)

output_list = []
    
for document in doc_list:
    output_dict = {}
#    print(document)
    if document["topic_num"] == "20":
                if document["year_bucket"] == "10":
                    output_dict["topic_no"] = document["topic_report"]["topic_no"]
                    output_dict["minstrel_value"] = document["topic_report"]["minstrel_value"]
                    output_dict["minstrel_z_score"] = document["topic_report"]["minstrel_z_score"]
                    use_list = []
                    for term in document["topic_report"]["topic_terms"]:
                        #for name, value in term.items():
                            print(term)
                            use_list.append(term["word"])
                    output_dict["top_words"] = use_list
                    output_list.append(output_dict)

for x in output_list:
    print(x)
