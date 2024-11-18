static int load_refcount_block(BlockDriverState *bs,

                               int64_t refcount_block_offset)

{

    BDRVQcowState *s = bs->opaque;

    int ret;



    if (cache_refcount_updates) {

        ret = write_refcount_block(bs);

        if (ret < 0) {

            return ret;

        }

    }



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_LOAD);

    ret = bdrv_pread(bs->file, refcount_block_offset, s->refcount_block_cache,

                     s->cluster_size);

    if (ret < 0) {


        return ret;

    }



    s->refcount_block_cache_offset = refcount_block_offset;

    return 0;

}