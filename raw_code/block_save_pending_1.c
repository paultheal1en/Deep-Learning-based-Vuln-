static uint64_t block_save_pending(QEMUFile *f, void *opaque, uint64_t max_size)

{

    /* Estimate pending number of bytes to send */

    uint64_t pending;



    qemu_mutex_lock_iothread();

    blk_mig_lock();

    pending = get_remaining_dirty() +

                       block_mig_state.submitted * BLOCK_SIZE +

                       block_mig_state.read_done * BLOCK_SIZE;



    /* Report at least one block pending during bulk phase */

    if (pending == 0 && !block_mig_state.bulk_completed) {

        pending = BLOCK_SIZE;

    }

    blk_mig_unlock();

    qemu_mutex_unlock_iothread();



    DPRINTF("Enter save live pending  %" PRIu64 "\n", pending);

    return pending;

}
