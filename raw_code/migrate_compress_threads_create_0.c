void migrate_compress_threads_create(void)

{

    int i, thread_count;



    if (!migrate_use_compression()) {

        return;

    }

    quit_comp_thread = false;

    compression_switch = true;

    thread_count = migrate_compress_threads();

    compress_threads = g_new0(QemuThread, thread_count);

    comp_param = g_new0(CompressParam, thread_count);

    comp_done_cond = g_new0(QemuCond, 1);

    comp_done_lock = g_new0(QemuMutex, 1);

    qemu_cond_init(comp_done_cond);

    qemu_mutex_init(comp_done_lock);

    for (i = 0; i < thread_count; i++) {

        /* com_param[i].file is just used as a dummy buffer to save data, set

         * it's ops to empty.

         */

        comp_param[i].file = qemu_fopen_ops(NULL, &empty_ops);

        comp_param[i].done = true;

        qemu_mutex_init(&comp_param[i].mutex);

        qemu_cond_init(&comp_param[i].cond);

        qemu_thread_create(compress_threads + i, "compress",

                           do_data_compress, comp_param + i,

                           QEMU_THREAD_JOINABLE);

    }

}
