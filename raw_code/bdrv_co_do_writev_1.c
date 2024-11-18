static int coroutine_fn bdrv_co_do_writev(BlockDriverState *bs,
    int64_t sector_num, int nb_sectors, QEMUIOVector *qiov)
{
    BlockDriver *drv = bs->drv;
    BdrvTrackedRequest req;
    int ret;
    if (!bs->drv) {
        return -ENOMEDIUM;
    if (bs->read_only) {
        return -EACCES;
    if (bdrv_check_request(bs, sector_num, nb_sectors)) {
        return -EIO;
    /* throttling disk write I/O */
    if (bs->io_limits_enabled) {
        bdrv_io_limits_intercept(bs, true, nb_sectors);
    tracked_request_begin(&req, bs, sector_num, nb_sectors, true);
    ret = drv->bdrv_co_writev(bs, sector_num, nb_sectors, qiov);
    if (bs->dirty_bitmap) {
        set_dirty_bitmap(bs, sector_num, nb_sectors, 1);
    if (bs->wr_highest_sector < sector_num + nb_sectors - 1) {
        bs->wr_highest_sector = sector_num + nb_sectors - 1;
    tracked_request_end(&req);
    return ret;