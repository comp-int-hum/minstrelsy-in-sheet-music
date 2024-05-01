import json

output_jsonl = []
counter = 0 
with open("data/json_metadata.jsonl", "r") as in_file:
    for x in in_file:
        if counter > 300:
            break 
        output_jsonl.append(json.loads(x))
        counter += 1
with open("data/toy_data_set.jsonl", "w") as out_file: 
    for x in output_jsonl:
        json_line = json.dumps(x)
        out_file.write(json_line + "\n")
    

