static int qcow_make_empty(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    uint32_t l1_length = s->l1_size * sizeof(uint64_t);

    int ret;



    memset(s->l1_table, 0, l1_length);

    if (bdrv_pwrite(bs->file, s->l1_table_offset, s->l1_table, l1_length) < 0)

	return -1;

    ret = bdrv_truncate(bs->file, s->l1_table_offset + l1_length);

    if (ret < 0)

        return ret;



    memset(s->l2_cache, 0, s->l2_size * L2_CACHE_SIZE * sizeof(uint64_t));

    memset(s->l2_cache_offsets, 0, L2_CACHE_SIZE * sizeof(uint64_t));

    memset(s->l2_cache_counts, 0, L2_CACHE_SIZE * sizeof(uint32_t));



    return 0;

}
