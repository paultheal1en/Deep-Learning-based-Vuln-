int qcow2_cache_put(BlockDriverState *bs, Qcow2Cache *c, void **table)

{

    int i = qcow2_cache_get_table_idx(bs, c, *table);



    if (c->entries[i].offset == 0) {

        return -ENOENT;

    }



    c->entries[i].ref--;

    *table = NULL;



    if (c->entries[i].ref == 0) {

        c->entries[i].lru_counter = ++c->lru_counter;

    }



    assert(c->entries[i].ref >= 0);

    return 0;

}
