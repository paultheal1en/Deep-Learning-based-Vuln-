static int qemu_event_init(void)

{

    int err;

    int fds[2];



    err = pipe(fds);

    if (err == -1)

        return -errno;



    err = fcntl_setfl(fds[0], O_NONBLOCK);

    if (err < 0)

        goto fail;



    err = fcntl_setfl(fds[1], O_NONBLOCK);

    if (err < 0)

        goto fail;



    qemu_set_fd_handler2(fds[0], NULL, qemu_event_read, NULL,

                         (void *)(unsigned long)fds[0]);



    io_thread_fd = fds[1];

    return 0;



fail:

    close(fds[0]);

    close(fds[1]);

    return err;

}
