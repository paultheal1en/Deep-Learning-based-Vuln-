static int coroutine_fn raw_co_writev(BlockDriverState *bs, int64_t sector_num,

                                      int nb_sectors, QEMUIOVector *qiov)

{

    BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

    return bdrv_co_writev(bs->file, sector_num, nb_sectors, qiov);

}
