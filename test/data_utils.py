import json


def __read_jsonl(file):
    lines = []
    with open(file) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def read_data(root, split):
    tables = __read_jsonl(root + "/" + split +".tables.jsonl")
    tables = {x["id"] : x for x in tables}
    queries = __read_jsonl(root + "/" + split + "_tok.jsonl")
    return queries, tables



if __name__ == "__main__":
    x = read_data("./data_root/clean", "dev")
    print(x[0])