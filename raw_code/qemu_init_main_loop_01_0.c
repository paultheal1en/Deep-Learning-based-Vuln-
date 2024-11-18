int qemu_init_main_loop(void)

{

    int ret;



    qemu_init_sigbus();



    ret = qemu_signal_init();

    if (ret) {

        return ret;

    }



    /* Note eventfd must be drained before signalfd handlers run */

    ret = qemu_event_init();

    if (ret) {

        return ret;

    }



    qemu_cond_init(&qemu_cpu_cond);

    qemu_cond_init(&qemu_system_cond);

    qemu_cond_init(&qemu_pause_cond);

    qemu_cond_init(&qemu_work_cond);

    qemu_cond_init(&qemu_io_proceeded_cond);

    qemu_mutex_init(&qemu_global_mutex);

    qemu_mutex_lock(&qemu_global_mutex);



    qemu_thread_get_self(&io_thread);



    return 0;

}
