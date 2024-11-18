static int posix_aio_init(void)

{

    struct sigaction act;

    PosixAioState *s;

    int fds[2];

    struct qemu_paioinit ai;

  

    if (posix_aio_state)

        return 0;



    s = qemu_malloc(sizeof(PosixAioState));



    sigfillset(&act.sa_mask);

    act.sa_flags = 0; /* do not restart syscalls to interrupt select() */

    act.sa_handler = aio_signal_handler;

    sigaction(SIGUSR2, &act, NULL);



    s->first_aio = NULL;

    if (pipe(fds) == -1) {

        fprintf(stderr, "failed to create pipe\n");

        return -errno;

    }



    s->rfd = fds[0];

    s->wfd = fds[1];



    fcntl(s->rfd, F_SETFL, O_NONBLOCK);

    fcntl(s->wfd, F_SETFL, O_NONBLOCK);



    qemu_aio_set_fd_handler(s->rfd, posix_aio_read, NULL, posix_aio_flush, s);



    memset(&ai, 0, sizeof(ai));

    ai.aio_threads = 64;

    ai.aio_num = 64;

    qemu_paio_init(&ai);



    posix_aio_state = s;



    return 0;

}
