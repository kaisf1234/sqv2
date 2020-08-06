import random

query_memo = {}
table_memo = {}

from flashtext import KeywordProcessor


def build_flash_table(samples):
    keyword_processor = KeywordProcessor(case_sensitive=False)
    keyword_processor.add_keywords_from_list([value for col in samples.values() for value in col])
    col_sets = {col : set(samples[col]) for col in samples}
    return (col_sets, keyword_processor)


def get_best_samples(query, samples, table_id):
    if (query, table_id) in query_memo:
        return query_memo[(query, table_id)]
    if table_id in table_memo:
        flash_table = table_memo[table_id]
    else:
        flash_table = build_flash_table(samples)
        table_memo[table_id] = flash_table
    col_sets, keyword_processor = flash_table
    keywords = keyword_processor.extract_keywords(query)
    best_samples = {name : set() for name in samples}
    for keyword in keywords:
        for header in col_sets:
            if keyword in col_sets[header]:
                best_samples[header].add(keyword)
    query_memo[(query, table_id)] = best_samples
    return best_samples

def pre_proc(batch, tables, column_samples):
    nlus = []
    sqls = []
    headers = []
    values = []
    nlu_origs = []
    for element in batch:
        nlu = element["question_tok"]
        nlu_orig = element["question"]
        sql = element["sql"]
        table = tables[element["table_id"]]
        header = table["header"]
        tid1 = table["id"]
        best_samples = get_best_samples(element["question"], column_samples[tid1], tid1)
        sampled_headers_values = {x:random.sample(best_samples[x], min(len(best_samples[x]), 3)) + [val for val in random.sample(set(column_samples[tid1][x]) - best_samples[x], min(len(set(column_samples[tid1][x]) - best_samples[x]), max(0, 3 - len(best_samples[x]))))]
                                      for x in table["header"]}
        _ = {x:random.shuffle(sampled_headers_values[x]) for x in sampled_headers_values}
        value = sampled_headers_values
        value = {x:random.sample(value[x], min(len(value[x]), 3)) + [value[x][0] for _ in range(3-len(value[x]))] for x in value}
        synthetic_rows = [[] for _ in range(3)]
        for header_name in value:
            for row_idx, cell in enumerate(value[header_name]):
                synthetic_rows[row_idx].append(cell)
        value = synthetic_rows

        nlus.append(nlu)
        sqls.append(sql)
        headers.append(header)
        values.append(value)
        nlu_origs.append(nlu_orig)
    return nlus, sqls, headers, values, nlu_origs