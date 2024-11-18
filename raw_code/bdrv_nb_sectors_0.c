int64_t bdrv_nb_sectors(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;



    if (!drv)

        return -ENOMEDIUM;



    if (drv->has_variable_length) {

        int ret = refresh_total_sectors(bs, bs->total_sectors);

        if (ret < 0) {

            return ret;

        }

    }

    return bs->total_sectors;

}
