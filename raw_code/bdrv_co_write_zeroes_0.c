int coroutine_fn bdrv_co_write_zeroes(BlockDriverState *bs,

                                      int64_t sector_num, int nb_sectors,

                                      BdrvRequestFlags flags)

{

    trace_bdrv_co_write_zeroes(bs, sector_num, nb_sectors, flags);



    if (!(bs->open_flags & BDRV_O_UNMAP)) {

        flags &= ~BDRV_REQ_MAY_UNMAP;

    }



    return bdrv_co_do_writev(bs, sector_num, nb_sectors, NULL,

                             BDRV_REQ_ZERO_WRITE | flags);

}
