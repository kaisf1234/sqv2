from table_bert import Table, Column


def tabert_tokenize_tables(headers, values, ids, tokenizer):
    tables = []
    for idx, header, value in zip(ids, headers, values):
        table = Table(
            id=idx,
            header=[
                Column(header_name, 'text', sample_value=value[0][header_idx]) for header_idx, header_name in enumerate(header)
            ],
            data=value
        ).tokenize(tokenizer)
        tables.append(table)
    return tables

def tabert_get_contexts(queries, tokenizer):
    return [tokenizer.tokenize(' '.join(x)) for x in queries]
