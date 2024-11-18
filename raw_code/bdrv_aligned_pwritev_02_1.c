static int coroutine_fn bdrv_aligned_pwritev(BlockDriverState *bs,

    BdrvTrackedRequest *req, int64_t offset, unsigned int bytes,

    QEMUIOVector *qiov, int flags)

{

    BlockDriver *drv = bs->drv;

    bool waited;

    int ret;



    int64_t sector_num = offset >> BDRV_SECTOR_BITS;

    unsigned int nb_sectors = bytes >> BDRV_SECTOR_BITS;



    assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert(!qiov || bytes == qiov->size);



    waited = wait_serialising_requests(req);

    assert(!waited || !req->serialising);

    assert(req->overlap_offset <= offset);

    assert(offset + bytes <= req->overlap_offset + req->overlap_bytes);



    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, req);



    if (!ret && bs->detect_zeroes != BLOCKDEV_DETECT_ZEROES_OPTIONS_OFF &&

        !(flags & BDRV_REQ_ZERO_WRITE) && drv->bdrv_co_write_zeroes &&

        qemu_iovec_is_zero(qiov)) {

        flags |= BDRV_REQ_ZERO_WRITE;

        if (bs->detect_zeroes == BLOCKDEV_DETECT_ZEROES_OPTIONS_UNMAP) {

            flags |= BDRV_REQ_MAY_UNMAP;

        }

    }



    if (ret < 0) {

        /* Do nothing, write notifier decided to fail this request */

    } else if (flags & BDRV_REQ_ZERO_WRITE) {

        BLKDBG_EVENT(bs, BLKDBG_PWRITEV_ZERO);

        ret = bdrv_co_do_write_zeroes(bs, sector_num, nb_sectors, flags);

    } else {

        BLKDBG_EVENT(bs, BLKDBG_PWRITEV);

        ret = drv->bdrv_co_writev(bs, sector_num, nb_sectors, qiov);

    }

    BLKDBG_EVENT(bs, BLKDBG_PWRITEV_DONE);



    if (ret == 0 && !bs->enable_write_cache) {

        ret = bdrv_co_flush(bs);

    }



    bdrv_set_dirty(bs, sector_num, nb_sectors);



    if (bs->stats.wr_highest_sector < sector_num + nb_sectors - 1) {

        bs->stats.wr_highest_sector = sector_num + nb_sectors - 1;

    }

    if (bs->growable && ret >= 0) {

        bs->total_sectors = MAX(bs->total_sectors, sector_num + nb_sectors);

    }



    return ret;

}
