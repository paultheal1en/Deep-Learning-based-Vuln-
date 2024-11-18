static int l2_load(BlockDriverState *bs, uint64_t l2_offset,

    uint64_t **l2_table)

{

    BDRVQcow2State *s = bs->opaque;

    int ret;



    ret = qcow2_cache_get(bs, s->l2_table_cache, l2_offset, (void**) l2_table);



    return ret;

}
