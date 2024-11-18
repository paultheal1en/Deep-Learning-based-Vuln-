static int qemu_signal_init(void)

{

    int sigfd;

    sigset_t set;



#ifdef CONFIG_IOTHREAD

    /* SIGUSR2 used by posix-aio-compat.c */

    sigemptyset(&set);

    sigaddset(&set, SIGUSR2);

    pthread_sigmask(SIG_UNBLOCK, &set, NULL);



    /*

     * SIG_IPI must be blocked in the main thread and must not be caught

     * by sigwait() in the signal thread. Otherwise, the cpu thread will

     * not catch it reliably.

     */

    sigemptyset(&set);

    sigaddset(&set, SIG_IPI);

    pthread_sigmask(SIG_BLOCK, &set, NULL);



    sigemptyset(&set);

    sigaddset(&set, SIGIO);

    sigaddset(&set, SIGALRM);

    sigaddset(&set, SIGBUS);

#else

    sigemptyset(&set);

    sigaddset(&set, SIGBUS);

    if (kvm_enabled()) {

        /*

         * We need to process timer signals synchronously to avoid a race

         * between exit_request check and KVM vcpu entry.

         */

        sigaddset(&set, SIGIO);

        sigaddset(&set, SIGALRM);

    }

#endif

    pthread_sigmask(SIG_BLOCK, &set, NULL);



    sigfd = qemu_signalfd(&set);

    if (sigfd == -1) {

        fprintf(stderr, "failed to create signalfd\n");

        return -errno;

    }



    fcntl_setfl(sigfd, O_NONBLOCK);



    qemu_set_fd_handler2(sigfd, NULL, sigfd_handler, NULL,

                         (void *)(intptr_t)sigfd);



    return 0;

}
