static int find_image_format(BlockDriverState *bs, const char *filename,

                             BlockDriver **pdrv, Error **errp)

{

    BlockDriver *drv;

    uint8_t buf[BLOCK_PROBE_BUF_SIZE];

    int ret = 0;



    /* Return the raw BlockDriver * to scsi-generic devices or empty drives */

    if (bs->sg || !bdrv_is_inserted(bs) || bdrv_getlength(bs) == 0) {

        *pdrv = &bdrv_raw;

        return ret;

    }



    ret = bdrv_pread(bs, 0, buf, sizeof(buf));

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not read image for determining its "

                         "format");

        *pdrv = NULL;

        return ret;

    }



    drv = bdrv_probe_all(buf, ret, filename);

    if (!drv) {

        error_setg(errp, "Could not determine image format: No compatible "

                   "driver found");

        ret = -ENOENT;

    }

    *pdrv = drv;

    return ret;

}
