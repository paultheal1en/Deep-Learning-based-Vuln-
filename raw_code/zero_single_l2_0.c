static int zero_single_l2(BlockDriverState *bs, uint64_t offset,

    unsigned int nb_clusters)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l2_table;

    int l2_index;

    int ret;

    int i;



    ret = get_cluster_table(bs, offset, &l2_table, &l2_index);

    if (ret < 0) {

        return ret;

    }



    /* Limit nb_clusters to one L2 table */

    nb_clusters = MIN(nb_clusters, s->l2_size - l2_index);



    for (i = 0; i < nb_clusters; i++) {

        uint64_t old_offset;



        old_offset = be64_to_cpu(l2_table[l2_index + i]);



        /* Update L2 entries */

        qcow2_cache_entry_mark_dirty(bs, s->l2_table_cache, l2_table);

        if (old_offset & QCOW_OFLAG_COMPRESSED) {

            l2_table[l2_index + i] = cpu_to_be64(QCOW_OFLAG_ZERO);

            qcow2_free_any_clusters(bs, old_offset, 1, QCOW2_DISCARD_REQUEST);

        } else {

            l2_table[l2_index + i] |= cpu_to_be64(QCOW_OFLAG_ZERO);

        }

    }



    ret = qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

    if (ret < 0) {

        return ret;

    }



    return nb_clusters;

}
