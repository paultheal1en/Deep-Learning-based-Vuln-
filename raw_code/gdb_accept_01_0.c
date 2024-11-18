static void gdb_accept(void)

{

    GDBState *s;

    struct sockaddr_in sockaddr;

    socklen_t len;

    int fd;



    for(;;) {

        len = sizeof(sockaddr);

        fd = accept(gdbserver_fd, (struct sockaddr *)&sockaddr, &len);

        if (fd < 0 && errno != EINTR) {

            perror("accept");

            return;

        } else if (fd >= 0) {

#ifndef _WIN32

            fcntl(fd, F_SETFD, FD_CLOEXEC);

#endif

            break;

        }

    }



    /* set short latency */

    socket_set_nodelay(fd);



    s = g_malloc0(sizeof(GDBState));

    s->c_cpu = first_cpu;

    s->g_cpu = first_cpu;

    s->fd = fd;

    gdb_has_xml = false;



    gdbserver_state = s;



    fcntl(fd, F_SETFL, O_NONBLOCK);

}
