static void flush_compressed_data(QEMUFile *f)

{

    int idx, len, thread_count;



    if (!migrate_use_compression()) {

        return;

    }

    thread_count = migrate_compress_threads();

    for (idx = 0; idx < thread_count; idx++) {

        if (!comp_param[idx].done) {

            qemu_mutex_lock(comp_done_lock);

            while (!comp_param[idx].done && !quit_comp_thread) {

                qemu_cond_wait(comp_done_cond, comp_done_lock);

            }

            qemu_mutex_unlock(comp_done_lock);

        }

        if (!quit_comp_thread) {

            len = qemu_put_qemu_file(f, comp_param[idx].file);

            bytes_transferred += len;

        }

    }

}
