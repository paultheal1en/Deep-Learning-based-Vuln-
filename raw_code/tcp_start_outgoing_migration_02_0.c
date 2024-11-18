MigrationState *tcp_start_outgoing_migration(const char *host_port,

					     int64_t bandwidth_limit,

					     int async)

{

    struct sockaddr_in addr;

    FdMigrationState *s;

    int ret;



    if (parse_host_port(&addr, host_port) < 0)

        return NULL;



    s = qemu_mallocz(sizeof(*s));

    if (s == NULL)

        return NULL;



    s->mig_state.cancel = tcp_cancel;

    s->mig_state.get_status = tcp_get_status;

    s->mig_state.release = tcp_release;



    s->state = MIG_STATE_ACTIVE;

    s->detach = !async;

    s->bandwidth_limit = bandwidth_limit;

    s->fd = socket(PF_INET, SOCK_STREAM, 0);

    if (s->fd == -1) {

        qemu_free(s);

	return NULL;

    }



    fcntl(s->fd, F_SETFL, O_NONBLOCK);



    if (s->detach == 1) {

        dprintf("detaching from monitor\n");

        monitor_suspend();

	s->detach = 2;

    }



    do {

        ret = connect(s->fd, (struct sockaddr *)&addr, sizeof(addr));

        if (ret == -1)

            ret = -errno;



        if (ret == -EINPROGRESS)

            qemu_set_fd_handler2(s->fd, NULL, NULL, tcp_wait_for_connect, s);

    } while (ret == -EINTR);



    if (ret < 0 && ret != -EINPROGRESS) {

        dprintf("connect failed\n");

        close(s->fd);

        qemu_free(s);

	s = NULL;

    } else if (ret >= 0)

        tcp_connect_migrate(s);



    return &s->mig_state;

}
