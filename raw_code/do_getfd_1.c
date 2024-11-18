static int do_getfd(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const char *fdname = qdict_get_str(qdict, "fdname");

    mon_fd_t *monfd;

    int fd;



    fd = qemu_chr_get_msgfd(mon->chr);

    if (fd == -1) {

        qerror_report(QERR_FD_NOT_SUPPLIED);

        return -1;

    }



    if (qemu_isdigit(fdname[0])) {

        qerror_report(QERR_INVALID_PARAMETER_VALUE, "fdname",

                      "a name not starting with a digit");

        return -1;

    }



    fd = dup(fd);

    if (fd == -1) {

        if (errno == EMFILE)

            qerror_report(QERR_TOO_MANY_FILES);

        else

            qerror_report(QERR_UNDEFINED_ERROR);

        return -1;

    }



    QLIST_FOREACH(monfd, &mon->fds, next) {

        if (strcmp(monfd->name, fdname) != 0) {

            continue;

        }



        close(monfd->fd);

        monfd->fd = fd;

        return 0;

    }



    monfd = qemu_mallocz(sizeof(mon_fd_t));

    monfd->name = qemu_strdup(fdname);

    monfd->fd = fd;



    QLIST_INSERT_HEAD(&mon->fds, monfd, next);

    return 0;

}
