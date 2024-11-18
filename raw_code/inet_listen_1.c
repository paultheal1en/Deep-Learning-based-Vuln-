int inet_listen(const char *str, char *ostr, int olen,

                int socktype, int port_offset, Error **errp)

{

    QemuOpts *opts;

    char *optstr;

    int sock = -1;



    opts = qemu_opts_create(&dummy_opts, NULL, 0);

    if (inet_parse(opts, str) == 0) {

        sock = inet_listen_opts(opts, port_offset, errp);

        if (sock != -1 && ostr) {

            optstr = strchr(str, ',');

            if (qemu_opt_get_bool(opts, "ipv6", 0)) {

                snprintf(ostr, olen, "[%s]:%s%s",

                         qemu_opt_get(opts, "host"),

                         qemu_opt_get(opts, "port"),

                         optstr ? optstr : "");

            } else {

                snprintf(ostr, olen, "%s:%s%s",

                         qemu_opt_get(opts, "host"),

                         qemu_opt_get(opts, "port"),

                         optstr ? optstr : "");

            }

        }

    } else {

        error_set(errp, QERR_SOCKET_CREATE_FAILED);

    }

    qemu_opts_del(opts);

    return sock;

}
