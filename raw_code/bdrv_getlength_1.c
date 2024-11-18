int64_t bdrv_getlength(BlockDriverState *bs)

{

    int64_t ret = bdrv_nb_sectors(bs);




    return ret < 0 ? ret : ret * BDRV_SECTOR_SIZE;

}