static BlockAIOCB *bdrv_aio_readv_em(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockCompletionFunc *cb, void *opaque)

{

    return bdrv_aio_rw_vector(bs, sector_num, qiov, nb_sectors, cb, opaque, 0);

}
