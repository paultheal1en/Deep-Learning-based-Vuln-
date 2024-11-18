void qemu_mutex_lock_iothread(void)

{

    if (!tcg_enabled()) {

        qemu_mutex_lock(&qemu_global_mutex);

    } else {

        iothread_requesting_mutex = true;

        if (qemu_mutex_trylock(&qemu_global_mutex)) {

            qemu_cpu_kick_thread(first_cpu);

            qemu_mutex_lock(&qemu_global_mutex);

        }

        iothread_requesting_mutex = false;

        qemu_cond_broadcast(&qemu_io_proceeded_cond);

    }

}
