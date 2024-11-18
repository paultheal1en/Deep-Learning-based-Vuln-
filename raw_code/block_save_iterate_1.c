static int block_save_iterate(QEMUFile *f, void *opaque)

{

    int ret;

    int64_t last_ftell = qemu_ftell(f);



    DPRINTF("Enter save live iterate submitted %d transferred %d\n",

            block_mig_state.submitted, block_mig_state.transferred);



    ret = flush_blks(f);

    if (ret) {

        return ret;

    }



    blk_mig_reset_dirty_cursor();



    /* control the rate of transfer */

    blk_mig_lock();

    while ((block_mig_state.submitted +

            block_mig_state.read_done) * BLOCK_SIZE <

           qemu_file_get_rate_limit(f)) {

        blk_mig_unlock();

        if (block_mig_state.bulk_completed == 0) {

            /* first finish the bulk phase */

            if (blk_mig_save_bulked_block(f) == 0) {

                /* finished saving bulk on all devices */

                block_mig_state.bulk_completed = 1;

            }

            ret = 0;

        } else {

            /* Always called with iothread lock taken for

             * simplicity, block_save_complete also calls it.

             */

            qemu_mutex_lock_iothread();

            ret = blk_mig_save_dirty_block(f, 1);

            qemu_mutex_unlock_iothread();

        }

        if (ret < 0) {

            return ret;

        }

        blk_mig_lock();

        if (ret != 0) {

            /* no more dirty blocks */

            break;

        }

    }

    blk_mig_unlock();



    ret = flush_blks(f);

    if (ret) {

        return ret;

    }



    qemu_put_be64(f, BLK_MIG_FLAG_EOS);

    return qemu_ftell(f) - last_ftell;

}
