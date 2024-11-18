static int load_refcount_block(BlockDriverState *bs,

                               int64_t refcount_block_offset,

                               void **refcount_block)

{

    BDRVQcow2State *s = bs->opaque;

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_LOAD);

    ret = qcow2_cache_get(bs, s->refcount_block_cache, refcount_block_offset,

        refcount_block);



    return ret;

}
