int bdrv_read(BlockDriverState *bs, int64_t sector_num,

              uint8_t *buf, int nb_sectors)

{

    BlockDriver *drv = bs->drv;



    if (!drv)

        return -ENOMEDIUM;





    if (drv->bdrv_pread) {

        int ret, len;

        len = nb_sectors * 512;

        ret = drv->bdrv_pread(bs, sector_num * 512, buf, len);

        if (ret < 0)

            return ret;

        else if (ret != len)

            return -EINVAL;

        else {

	    bs->rd_bytes += (unsigned) len;

	    bs->rd_ops ++;

            return 0;

	}

    } else {

        return drv->bdrv_read(bs, sector_num, buf, nb_sectors);

    }

}