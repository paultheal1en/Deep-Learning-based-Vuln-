static int blk_mig_save_dirty_block(Monitor *mon, QEMUFile *f, int is_async)

{

    BlkMigDevState *bmds;

    int ret = 0;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        if (mig_save_device_dirty(mon, f, bmds, is_async) == 0) {

            ret = 1;

            break;

        }

    }



    return ret;

}
