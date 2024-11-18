static int ram_save_complete(QEMUFile *f, void *opaque)

{

    rcu_read_lock();



    migration_bitmap_sync();



    ram_control_before_iterate(f, RAM_CONTROL_FINISH);



    /* try transferring iterative blocks of memory */



    /* flush all remaining blocks regardless of rate limiting */

    while (true) {

        int pages;



        pages = ram_find_and_save_block(f, true, &bytes_transferred);

        /* no more blocks to sent */

        if (pages == 0) {

            break;

        }

    }



    flush_compressed_data(f);

    ram_control_after_iterate(f, RAM_CONTROL_FINISH);

    migration_end();



    rcu_read_unlock();

    qemu_put_be64(f, RAM_SAVE_FLAG_EOS);



    return 0;

}
