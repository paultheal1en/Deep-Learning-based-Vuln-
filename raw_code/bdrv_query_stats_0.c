static BlockStats *bdrv_query_stats(BlockBackend *blk,

                                    const BlockDriverState *bs,

                                    bool query_backing)

{

    BlockStats *s;



    s = bdrv_query_bds_stats(bs, query_backing);



    if (blk) {

        s->has_device = true;

        s->device = g_strdup(blk_name(blk));

        bdrv_query_blk_stats(s->stats, blk);

    }



    return s;

}
