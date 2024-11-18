static int parse_add_fd(QemuOpts *opts, void *opaque)

{

    int fd, dupfd, flags;

    int64_t fdset_id;

    const char *fd_opaque = NULL;



    fd = qemu_opt_get_number(opts, "fd", -1);

    fdset_id = qemu_opt_get_number(opts, "set", -1);

    fd_opaque = qemu_opt_get(opts, "opaque");



    if (fd < 0) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR,

                      "fd option is required and must be non-negative");

        return -1;

    }



    if (fd <= STDERR_FILENO) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR,

                      "fd cannot be a standard I/O stream");

        return -1;

    }



    /*

     * All fds inherited across exec() necessarily have FD_CLOEXEC

     * clear, while qemu sets FD_CLOEXEC on all other fds used internally.

     */

    flags = fcntl(fd, F_GETFD);

    if (flags == -1 || (flags & FD_CLOEXEC)) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR,

                      "fd is not valid or already in use");

        return -1;

    }



    if (fdset_id < 0) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR,

                      "set option is required and must be non-negative");

        return -1;

    }



#ifdef F_DUPFD_CLOEXEC

    dupfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);

#else

    dupfd = dup(fd);

    if (dupfd != -1) {

        qemu_set_cloexec(dupfd);

    }

#endif

    if (dupfd == -1) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR,

                      "Error duplicating fd: %s", strerror(errno));

        return -1;

    }



    /* add the duplicate fd, and optionally the opaque string, to the fd set */

    monitor_fdset_add_fd(dupfd, true, fdset_id, fd_opaque ? true : false,

                         fd_opaque, NULL);



    return 0;

}
