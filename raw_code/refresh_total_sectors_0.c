static int refresh_total_sectors(BlockDriverState *bs, int64_t hint)

{

    BlockDriver *drv = bs->drv;



    /* Do not attempt drv->bdrv_getlength() on scsi-generic devices */

    if (bs->sg)

        return 0;



    /* query actual device if possible, otherwise just trust the hint */

    if (drv->bdrv_getlength) {

        int64_t length = drv->bdrv_getlength(bs);

        if (length < 0) {

            return length;

        }

        hint = DIV_ROUND_UP(length, BDRV_SECTOR_SIZE);

    }



    bs->total_sectors = hint;

    return 0;

}
