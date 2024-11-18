int unix_listen_opts(QemuOpts *opts, Error **errp)

{

    struct sockaddr_un un;

    const char *path = qemu_opt_get(opts, "path");

    int sock, fd;



    sock = qemu_socket(PF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {

        error_setg_errno(errp, errno, "Failed to create Unix socket");

        return -1;

    }



    memset(&un, 0, sizeof(un));

    un.sun_family = AF_UNIX;

    if (path && strlen(path)) {

        snprintf(un.sun_path, sizeof(un.sun_path), "%s", path);

    } else {

        const char *tmpdir = getenv("TMPDIR");

        tmpdir = tmpdir ? tmpdir : "/tmp";

        if (snprintf(un.sun_path, sizeof(un.sun_path), "%s/qemu-socket-XXXXXX",

                     tmpdir) >= sizeof(un.sun_path)) {

            error_setg_errno(errp, errno,

                             "TMPDIR environment variable (%s) too large", tmpdir);

            goto err;

        }



        /*

         * This dummy fd usage silences the mktemp() unsecure warning.

         * Using mkstemp() doesn't make things more secure here

         * though.  bind() complains about existing files, so we have

         * to unlink first and thus re-open the race window.  The

         * worst case possible is bind() failing, i.e. a DoS attack.

         */

        fd = mkstemp(un.sun_path);

        if (fd < 0) {

            error_setg_errno(errp, errno,

                             "Failed to make a temporary socket name in %s", tmpdir);

            goto err;

        }

        close(fd);

        qemu_opt_set(opts, "path", un.sun_path, &error_abort);

    }



    if ((access(un.sun_path, F_OK) == 0) &&

        unlink(un.sun_path) < 0) {

        error_setg_errno(errp, errno,

                         "Failed to unlink socket %s", un.sun_path);

        goto err;

    }

    if (bind(sock, (struct sockaddr*) &un, sizeof(un)) < 0) {

        error_setg_errno(errp, errno, "Failed to bind socket to %s", un.sun_path);

        goto err;

    }

    if (listen(sock, 1) < 0) {

        error_setg_errno(errp, errno, "Failed to listen on socket");

        goto err;

    }



    return sock;



err:

    closesocket(sock);

    return -1;

}
