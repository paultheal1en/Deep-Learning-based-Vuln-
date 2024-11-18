void *laio_init(void)

{

    struct qemu_laio_state *s;



    s = qemu_mallocz(sizeof(*s));

    QLIST_INIT(&s->completed_reqs);

    s->efd = eventfd(0, 0);

    if (s->efd == -1)

        goto out_free_state;

    fcntl(s->efd, F_SETFL, O_NONBLOCK);



    if (io_setup(MAX_EVENTS, &s->ctx) != 0)

        goto out_close_efd;



    qemu_aio_set_fd_handler(s->efd, qemu_laio_completion_cb, NULL,

        qemu_laio_flush_cb, qemu_laio_process_requests, s);



    return s;



out_close_efd:

    close(s->efd);

out_free_state:

    qemu_free(s);

    return NULL;

}
