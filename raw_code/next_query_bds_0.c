static bool next_query_bds(BlockBackend **blk, BlockDriverState **bs,

                           bool query_nodes)

{

    if (query_nodes) {

        *bs = bdrv_next_node(*bs);

        return !!*bs;

    }



    *blk = blk_next(*blk);

    *bs = *blk ? blk_bs(*blk) : NULL;



    return !!*blk;

}
