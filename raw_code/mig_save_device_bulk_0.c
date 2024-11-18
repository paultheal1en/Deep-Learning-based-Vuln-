static int mig_save_device_bulk(Monitor *mon, QEMUFile *f,

                                BlkMigDevState *bmds)

{

    int64_t total_sectors = bmds->total_sectors;

    int64_t cur_sector = bmds->cur_sector;

    BlockDriverState *bs = bmds->bs;

    BlkMigBlock *blk;

    int nr_sectors;



    if (bmds->shared_base) {

        while (cur_sector < total_sectors &&

               !bdrv_is_allocated(bs, cur_sector, MAX_IS_ALLOCATED_SEARCH,

                                  &nr_sectors)) {

            cur_sector += nr_sectors;

        }

    }



    if (cur_sector >= total_sectors) {

        bmds->cur_sector = bmds->completed_sectors = total_sectors;

        return 1;

    }



    bmds->completed_sectors = cur_sector;



    cur_sector &= ~((int64_t)BDRV_SECTORS_PER_DIRTY_CHUNK - 1);



    /* we are going to transfer a full block even if it is not allocated */

    nr_sectors = BDRV_SECTORS_PER_DIRTY_CHUNK;



    if (total_sectors - cur_sector < BDRV_SECTORS_PER_DIRTY_CHUNK) {

        nr_sectors = total_sectors - cur_sector;

    }



    blk = g_malloc(sizeof(BlkMigBlock));

    blk->buf = g_malloc(BLOCK_SIZE);

    blk->bmds = bmds;

    blk->sector = cur_sector;

    blk->nr_sectors = nr_sectors;



    blk->iov.iov_base = blk->buf;

    blk->iov.iov_len = nr_sectors * BDRV_SECTOR_SIZE;

    qemu_iovec_init_external(&blk->qiov, &blk->iov, 1);



    if (block_mig_state.submitted == 0) {

        block_mig_state.prev_time_offset = qemu_get_clock_ns(rt_clock);

    }



    blk->aiocb = bdrv_aio_readv(bs, cur_sector, &blk->qiov,

                                nr_sectors, blk_mig_read_cb, blk);

    block_mig_state.submitted++;



    bdrv_reset_dirty(bs, cur_sector, nr_sectors);

    bmds->cur_sector = cur_sector + nr_sectors;



    return (bmds->cur_sector >= total_sectors);

}
