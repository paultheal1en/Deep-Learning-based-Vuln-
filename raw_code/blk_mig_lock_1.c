static void blk_mig_lock(void)

{

    qemu_mutex_lock(&block_mig_state.lock);

}
