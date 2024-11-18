static void qemu_chr_parse_socket(QemuOpts *opts, ChardevBackend *backend,

                                  Error **errp)

{

    bool is_listen      = qemu_opt_get_bool(opts, "server", false);

    bool is_waitconnect = is_listen && qemu_opt_get_bool(opts, "wait", true);

    bool is_telnet      = qemu_opt_get_bool(opts, "telnet", false);

    bool is_tn3270      = qemu_opt_get_bool(opts, "tn3270", false);

    bool do_nodelay     = !qemu_opt_get_bool(opts, "delay", true);

    int64_t reconnect   = qemu_opt_get_number(opts, "reconnect", 0);

    const char *path = qemu_opt_get(opts, "path");

    const char *host = qemu_opt_get(opts, "host");

    const char *port = qemu_opt_get(opts, "port");

    const char *tls_creds = qemu_opt_get(opts, "tls-creds");

    SocketAddress *addr;

    ChardevSocket *sock;



    backend->type = CHARDEV_BACKEND_KIND_SOCKET;

    if (!path) {

        if (!host) {

            error_setg(errp, "chardev: socket: no host given");

            return;

        }

        if (!port) {

            error_setg(errp, "chardev: socket: no port given");

            return;

        }

    } else {

        if (tls_creds) {

            error_setg(errp, "TLS can only be used over TCP socket");

            return;

        }

    }



    sock = backend->u.socket.data = g_new0(ChardevSocket, 1);

    qemu_chr_parse_common(opts, qapi_ChardevSocket_base(sock));



    sock->has_nodelay = true;

    sock->nodelay = do_nodelay;

    sock->has_server = true;

    sock->server = is_listen;

    sock->has_telnet = true;

    sock->telnet = is_telnet;

    sock->has_tn3270 = true;

    sock->tn3270 = is_tn3270;

    sock->has_wait = true;

    sock->wait = is_waitconnect;

    sock->has_reconnect = true;

    sock->reconnect = reconnect;

    sock->tls_creds = g_strdup(tls_creds);



    addr = g_new0(SocketAddress, 1);

    if (path) {

        UnixSocketAddress *q_unix;

        addr->type = SOCKET_ADDRESS_KIND_UNIX;

        q_unix = addr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

        q_unix->path = g_strdup(path);

    } else {

        addr->type = SOCKET_ADDRESS_KIND_INET;

        addr->u.inet.data = g_new(InetSocketAddress, 1);

        *addr->u.inet.data = (InetSocketAddress) {

            .host = g_strdup(host),

            .port = g_strdup(port),

            .has_to = qemu_opt_get(opts, "to"),

            .to = qemu_opt_get_number(opts, "to", 0),

            .has_ipv4 = qemu_opt_get(opts, "ipv4"),

            .ipv4 = qemu_opt_get_bool(opts, "ipv4", 0),

            .has_ipv6 = qemu_opt_get(opts, "ipv6"),

            .ipv6 = qemu_opt_get_bool(opts, "ipv6", 0),

        };

    }

    sock->addr = addr;

}
