static SocketAddress *nbd_config(BDRVNBDState *s, QDict *options, char **export,

                                 Error **errp)

{

    SocketAddress *saddr;



    if (qdict_haskey(options, "path") == qdict_haskey(options, "host")) {

        if (qdict_haskey(options, "path")) {

            error_setg(errp, "path and host may not be used at the same time.");

        } else {

            error_setg(errp, "one of path and host must be specified.");

        }

        return NULL;

    }



    saddr = g_new0(SocketAddress, 1);



    if (qdict_haskey(options, "path")) {

        UnixSocketAddress *q_unix;

        saddr->type = SOCKET_ADDRESS_KIND_UNIX;

        q_unix = saddr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

        q_unix->path = g_strdup(qdict_get_str(options, "path"));

        qdict_del(options, "path");

    } else {

        InetSocketAddress *inet;

        saddr->type = SOCKET_ADDRESS_KIND_INET;

        inet = saddr->u.inet.data = g_new0(InetSocketAddress, 1);

        inet->host = g_strdup(qdict_get_str(options, "host"));

        if (!qdict_get_try_str(options, "port")) {

            inet->port = g_strdup_printf("%d", NBD_DEFAULT_PORT);

        } else {

            inet->port = g_strdup(qdict_get_str(options, "port"));

        }

        qdict_del(options, "host");

        qdict_del(options, "port");

    }



    s->client.is_unix = saddr->type == SOCKET_ADDRESS_KIND_UNIX;



    *export = g_strdup(qdict_get_try_str(options, "export"));

    if (*export) {

        qdict_del(options, "export");

    }



    return saddr;

}
