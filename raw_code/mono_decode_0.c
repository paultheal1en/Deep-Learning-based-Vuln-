static int mono_decode(COOKContext *q, COOKSubpacket *p, float *mlt_buffer)

{

    int category_index[128];

    int quant_index_table[102];

    int category[128];

    int ret;



    memset(&category,       0, sizeof(category));

    memset(&category_index, 0, sizeof(category_index));



    if ((ret = decode_envelope(q, p, quant_index_table)) < 0)

        return ret;

    q->num_vectors = get_bits(&q->gb, p->log2_numvector_size);

    categorize(q, p, quant_index_table, category, category_index);

    expand_category(q, category, category_index);

    decode_vectors(q, p, category, quant_index_table, mlt_buffer);



    return 0;

}
