static void blk_mig_cleanup(Monitor *mon)

{

    BlkMigDevState *bmds;

    BlkMigBlock *blk;



    set_dirty_tracking(0);



    while ((bmds = QSIMPLEQ_FIRST(&block_mig_state.bmds_list)) != NULL) {

        QSIMPLEQ_REMOVE_HEAD(&block_mig_state.bmds_list, entry);

        bdrv_set_in_use(bmds->bs, 0);

        drive_put_ref(drive_get_by_blockdev(bmds->bs));

        g_free(bmds->aio_bitmap);

        g_free(bmds);

    }



    while ((blk = QSIMPLEQ_FIRST(&block_mig_state.blk_list)) != NULL) {

        QSIMPLEQ_REMOVE_HEAD(&block_mig_state.blk_list, entry);

        g_free(blk->buf);

        g_free(blk);

    }



    monitor_printf(mon, "\n");

}
