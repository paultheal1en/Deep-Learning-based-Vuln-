static void blk_mig_read_cb(void *opaque, int ret)

{

    BlkMigBlock *blk = opaque;



    blk_mig_lock();

    blk->ret = ret;



    QSIMPLEQ_INSERT_TAIL(&block_mig_state.blk_list, blk, entry);

    bmds_set_aio_inflight(blk->bmds, blk->sector, blk->nr_sectors, 0);



    block_mig_state.submitted--;

    block_mig_state.read_done++;

    assert(block_mig_state.submitted >= 0);

    blk_mig_unlock();

}
