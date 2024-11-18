static int blk_mig_save_dirty_block(QEMUFile *f, int is_async)

{

    BlkMigDevState *bmds;

    int ret = 1;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        ret = mig_save_device_dirty(f, bmds, is_async);

        if (ret <= 0) {

            break;

        }

    }



    return ret;

}
