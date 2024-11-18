int bdrv_write_zeroes(BlockDriverState *bs, int64_t sector_num,

                      int nb_sectors, BdrvRequestFlags flags)

{

    return bdrv_rw_co(bs, sector_num, NULL, nb_sectors, true,

                      BDRV_REQ_ZERO_WRITE | flags);

}
