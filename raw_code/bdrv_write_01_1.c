int bdrv_write(BlockDriverState *bs, int64_t sector_num,
               const uint8_t *buf, int nb_sectors)
{
    BlockDriver *drv = bs->drv;
    if (!bs->drv)
        return -ENOMEDIUM;
    if (bs->read_only)
        return -EACCES;
    if (drv->bdrv_pwrite) {
        int ret, len, count = 0;
        len = nb_sectors * 512;
        do {
            ret = drv->bdrv_pwrite(bs, sector_num * 512, buf, len - count);
            if (ret < 0) {
                printf("bdrv_write ret=%d\n", ret);
                return ret;
            }
            count += ret;
            buf += ret;
        } while (count != len);
        bs->wr_bytes += (unsigned) len;
        bs->wr_ops ++;
        return 0;
    }
    return drv->bdrv_write(bs, sector_num, buf, nb_sectors);
}