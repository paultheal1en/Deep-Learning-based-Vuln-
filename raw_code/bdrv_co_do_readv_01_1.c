static int coroutine_fn bdrv_co_do_readv(BlockDriverState *bs,
    int64_t sector_num, int nb_sectors, QEMUIOVector *qiov)
{
    BlockDriver *drv = bs->drv;
    BdrvTrackedRequest req;
    int ret;
    if (!drv) {
        return -ENOMEDIUM;
    if (bdrv_check_request(bs, sector_num, nb_sectors)) {
        return -EIO;
    /* throttling disk read I/O */
    if (bs->io_limits_enabled) {
        bdrv_io_limits_intercept(bs, false, nb_sectors);
    tracked_request_begin(&req, bs, sector_num, nb_sectors, false);
    ret = drv->bdrv_co_readv(bs, sector_num, nb_sectors, qiov);
    tracked_request_end(&req);
    return ret;