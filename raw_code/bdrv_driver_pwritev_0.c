static int coroutine_fn bdrv_driver_pwritev(BlockDriverState *bs,

                                            uint64_t offset, uint64_t bytes,

                                            QEMUIOVector *qiov, int flags)

{

    BlockDriver *drv = bs->drv;

    int64_t sector_num = offset >> BDRV_SECTOR_BITS;

    unsigned int nb_sectors = bytes >> BDRV_SECTOR_BITS;

    int ret;



    assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes >> BDRV_SECTOR_BITS) <= BDRV_REQUEST_MAX_SECTORS);



    if (drv->bdrv_co_writev_flags) {

        ret = drv->bdrv_co_writev_flags(bs, sector_num, nb_sectors, qiov,

                                        flags);

    } else {

        assert(drv->supported_write_flags == 0);

        ret = drv->bdrv_co_writev(bs, sector_num, nb_sectors, qiov);

    }



    if (ret == 0 && (flags & BDRV_REQ_FUA) &&

        !(drv->supported_write_flags & BDRV_REQ_FUA))

    {

        ret = bdrv_co_flush(bs);

    }



    return ret;

}
