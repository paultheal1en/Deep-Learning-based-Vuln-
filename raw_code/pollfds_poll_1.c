static void pollfds_poll(GArray *pollfds, int nfds, fd_set *rfds,

                         fd_set *wfds, fd_set *xfds)

{

    int i;



    for (i = 0; i < pollfds->len; i++) {

        GPollFD *pfd = &g_array_index(pollfds, GPollFD, i);

        int fd = pfd->fd;

        int revents = 0;



        if (FD_ISSET(fd, rfds)) {

            revents |= G_IO_IN | G_IO_HUP | G_IO_ERR;

        }

        if (FD_ISSET(fd, wfds)) {

            revents |= G_IO_OUT | G_IO_ERR;

        }

        if (FD_ISSET(fd, xfds)) {

            revents |= G_IO_PRI;

        }

        pfd->revents = revents & pfd->events;

    }

}
