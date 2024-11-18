int coroutine_fn bdrv_co_writev(BlockDriverState *bs, int64_t sector_num,

    int nb_sectors, QEMUIOVector *qiov)

{

    trace_bdrv_co_writev(bs, sector_num, nb_sectors);



    return bdrv_co_do_writev(bs, sector_num, nb_sectors, qiov, 0);

}
