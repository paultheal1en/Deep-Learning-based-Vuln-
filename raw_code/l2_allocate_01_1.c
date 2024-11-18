static uint64_t *l2_allocate(BlockDriverState *bs, int l1_index)

{

    BDRVQcowState *s = bs->opaque;

    int min_index;

    uint64_t old_l2_offset;

    uint64_t *l2_table, l2_offset;



    old_l2_offset = s->l1_table[l1_index];



    /* allocate a new l2 entry */



    l2_offset = qcow2_alloc_clusters(bs, s->l2_size * sizeof(uint64_t));

    if (l2_offset < 0) {

        return NULL;

    }



    /* update the L1 entry */



    s->l1_table[l1_index] = l2_offset | QCOW_OFLAG_COPIED;

    if (write_l1_entry(s, l1_index) < 0) {

        return NULL;

    }



    /* allocate a new entry in the l2 cache */



    min_index = l2_cache_new_entry(bs);

    l2_table = s->l2_cache + (min_index << s->l2_bits);



    if (old_l2_offset == 0) {

        /* if there was no old l2 table, clear the new table */

        memset(l2_table, 0, s->l2_size * sizeof(uint64_t));

    } else {

        /* if there was an old l2 table, read it from the disk */

        if (bdrv_pread(s->hd, old_l2_offset,

                       l2_table, s->l2_size * sizeof(uint64_t)) !=

            s->l2_size * sizeof(uint64_t))

            return NULL;

    }

    /* write the l2 table to the file */

    if (bdrv_pwrite(s->hd, l2_offset,

                    l2_table, s->l2_size * sizeof(uint64_t)) !=

        s->l2_size * sizeof(uint64_t))

        return NULL;



    /* update the l2 cache entry */



    s->l2_cache_offsets[min_index] = l2_offset;

    s->l2_cache_counts[min_index] = 1;



    return l2_table;

}
