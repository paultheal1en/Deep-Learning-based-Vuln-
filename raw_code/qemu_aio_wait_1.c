void qemu_aio_wait(void)

{

    sigset_t set;

    int nb_sigs;



#if !defined(QEMU_IMG) && !defined(QEMU_NBD)

    if (qemu_bh_poll())

        return;

#endif

    sigemptyset(&set);

    sigaddset(&set, aio_sig_num);

    sigwait(&set, &nb_sigs);

    qemu_aio_poll();

}
