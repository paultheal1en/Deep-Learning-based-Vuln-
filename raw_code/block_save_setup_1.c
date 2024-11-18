static int block_save_setup(QEMUFile *f, void *opaque)

{

    int ret;



    DPRINTF("Enter save live setup submitted %d transferred %d\n",

            block_mig_state.submitted, block_mig_state.transferred);



    qemu_mutex_lock_iothread();

    init_blk_migration(f);



    /* start track dirty blocks */

    set_dirty_tracking();

    qemu_mutex_unlock_iothread();



    ret = flush_blks(f);

    blk_mig_reset_dirty_cursor();

    qemu_put_be64(f, BLK_MIG_FLAG_EOS);



    return ret;

}
