static int qemu_signal_init(void)

{

    int sigfd;

    sigset_t set;



    /*

     * SIG_IPI must be blocked in the main thread and must not be caught

     * by sigwait() in the signal thread. Otherwise, the cpu thread will

     * not catch it reliably.

     */

    sigemptyset(&set);

    sigaddset(&set, SIG_IPI);

    sigaddset(&set, SIGIO);

    sigaddset(&set, SIGALRM);

    sigaddset(&set, SIGBUS);

    sigaddset(&set, SIGINT);

    sigaddset(&set, SIGHUP);

    sigaddset(&set, SIGTERM);

    pthread_sigmask(SIG_BLOCK, &set, NULL);



    sigdelset(&set, SIG_IPI);

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
