static void do_getfd(Monitor *mon, const QDict *qdict)

{

    const char *fdname = qdict_get_str(qdict, "fdname");

    mon_fd_t *monfd;

    int fd;



    fd = qemu_chr_get_msgfd(mon->chr);

    if (fd == -1) {

        monitor_printf(mon, "getfd: no file descriptor supplied via SCM_RIGHTS\n");

        return;

    }



    if (qemu_isdigit(fdname[0])) {

        monitor_printf(mon, "getfd: monitor names may not begin with a number\n");

        return;

    }



    fd = dup(fd);

    if (fd == -1) {

        monitor_printf(mon, "Failed to dup() file descriptor: %s\n",

                       strerror(errno));

        return;

    }



    LIST_FOREACH(monfd, &mon->fds, next) {

        if (strcmp(monfd->name, fdname) != 0) {

            continue;

        }



        close(monfd->fd);

        monfd->fd = fd;

        return;

    }



    monfd = qemu_mallocz(sizeof(mon_fd_t));

    monfd->name = qemu_strdup(fdname);

    monfd->fd = fd;



    LIST_INSERT_HEAD(&mon->fds, monfd, next);

}
