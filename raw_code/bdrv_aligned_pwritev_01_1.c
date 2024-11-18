static int coroutine_fn bdrv_aligned_pwritev(BdrvChild *child,
    BdrvTrackedRequest *req, int64_t offset, unsigned int bytes,
    int64_t align, QEMUIOVector *qiov, int flags)
{
    BlockDriverState *bs = child->bs;
    BlockDriver *drv = bs->drv;
    bool waited;
    int ret;
    int64_t end_sector = DIV_ROUND_UP(offset + bytes, BDRV_SECTOR_SIZE);
    uint64_t bytes_remaining = bytes;
    int max_transfer;
    if (bdrv_has_readonly_bitmaps(bs)) {
        return -EPERM;
    assert(is_power_of_2(align));
    assert((offset & (align - 1)) == 0);
    assert((bytes & (align - 1)) == 0);
    assert(!qiov || bytes == qiov->size);
    assert((bs->open_flags & BDRV_O_NO_IO) == 0);
    assert(!(flags & ~BDRV_REQ_MASK));
    max_transfer = QEMU_ALIGN_DOWN(MIN_NON_ZERO(bs->bl.max_transfer, INT_MAX),
                                   align);
    waited = wait_serialising_requests(req);
    assert(!waited || !req->serialising);
    assert(req->overlap_offset <= offset);
    assert(offset + bytes <= req->overlap_offset + req->overlap_bytes);
    assert(child->perm & BLK_PERM_WRITE);
    assert(end_sector <= bs->total_sectors || child->perm & BLK_PERM_RESIZE);
    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, req);
    if (!ret && bs->detect_zeroes != BLOCKDEV_DETECT_ZEROES_OPTIONS_OFF &&
        !(flags & BDRV_REQ_ZERO_WRITE) && drv->bdrv_co_pwrite_zeroes &&
        qemu_iovec_is_zero(qiov)) {
        flags |= BDRV_REQ_ZERO_WRITE;
        if (bs->detect_zeroes == BLOCKDEV_DETECT_ZEROES_OPTIONS_UNMAP) {
            flags |= BDRV_REQ_MAY_UNMAP;
    if (ret < 0) {
        /* Do nothing, write notifier decided to fail this request */
    } else if (flags & BDRV_REQ_ZERO_WRITE) {
        bdrv_debug_event(bs, BLKDBG_PWRITEV_ZERO);
        ret = bdrv_co_do_pwrite_zeroes(bs, offset, bytes, flags);
    } else if (flags & BDRV_REQ_WRITE_COMPRESSED) {
        ret = bdrv_driver_pwritev_compressed(bs, offset, bytes, qiov);
    } else if (bytes <= max_transfer) {
        bdrv_debug_event(bs, BLKDBG_PWRITEV);
        ret = bdrv_driver_pwritev(bs, offset, bytes, qiov, flags);
    } else {
        bdrv_debug_event(bs, BLKDBG_PWRITEV);
        while (bytes_remaining) {
            int num = MIN(bytes_remaining, max_transfer);
            QEMUIOVector local_qiov;
            int local_flags = flags;
            assert(num);
            if (num < bytes_remaining && (flags & BDRV_REQ_FUA) &&
                !(bs->supported_write_flags & BDRV_REQ_FUA)) {
                /* If FUA is going to be emulated by flush, we only
                 * need to flush on the last iteration */
                local_flags &= ~BDRV_REQ_FUA;
            qemu_iovec_init(&local_qiov, qiov->niov);
            qemu_iovec_concat(&local_qiov, qiov, bytes - bytes_remaining, num);
            ret = bdrv_driver_pwritev(bs, offset + bytes - bytes_remaining,
                                      num, &local_qiov, local_flags);
            qemu_iovec_destroy(&local_qiov);
            if (ret < 0) {
                break;
            bytes_remaining -= num;
    bdrv_debug_event(bs, BLKDBG_PWRITEV_DONE);
    atomic_inc(&bs->write_gen);
    bdrv_set_dirty(bs, offset, bytes);
    stat64_max(&bs->wr_highest_offset, offset + bytes);
    if (ret >= 0) {
        bs->total_sectors = MAX(bs->total_sectors, end_sector);
        ret = 0;
    return ret;