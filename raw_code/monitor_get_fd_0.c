int monitor_get_fd(Monitor *mon, const char *fdname)

{

    mon_fd_t *monfd;



    LIST_FOREACH(monfd, &mon->fds, next) {

        int fd;



        if (strcmp(monfd->name, fdname) != 0) {

            continue;

        }



        fd = monfd->fd;



        /* caller takes ownership of fd */

        LIST_REMOVE(monfd, next);

        qemu_free(monfd->name);

        qemu_free(monfd);



        return fd;

    }



    return -1;

}
