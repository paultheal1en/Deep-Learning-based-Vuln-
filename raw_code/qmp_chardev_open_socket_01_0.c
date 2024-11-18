static void qmp_chardev_open_socket(Chardev *chr,

                                    ChardevBackend *backend,

                                    bool *be_opened,

                                    Error **errp)

{

    SocketChardev *s = SOCKET_CHARDEV(chr);

    ChardevSocket *sock = backend->u.socket.data;

    SocketAddress *addr = sock->addr;

    bool do_nodelay     = sock->has_nodelay ? sock->nodelay : false;

    bool is_listen      = sock->has_server  ? sock->server  : true;

    bool is_telnet      = sock->has_telnet  ? sock->telnet  : false;

    bool is_tn3270      = sock->has_tn3270  ? sock->tn3270  : false;

    bool is_waitconnect = sock->has_wait    ? sock->wait    : false;

    int64_t reconnect   = sock->has_reconnect ? sock->reconnect : 0;

    QIOChannelSocket *sioc = NULL;



    s->is_listen = is_listen;

    s->is_telnet = is_telnet;

    s->is_tn3270 = is_tn3270;

    s->do_nodelay = do_nodelay;

    if (sock->tls_creds) {

        Object *creds;

        creds = object_resolve_path_component(

            object_get_objects_root(), sock->tls_creds);

        if (!creds) {

            error_setg(errp, "No TLS credentials with id '%s'",

                       sock->tls_creds);

            goto error;

        }

        s->tls_creds = (QCryptoTLSCreds *)

            object_dynamic_cast(creds,

                                TYPE_QCRYPTO_TLS_CREDS);

        if (!s->tls_creds) {

            error_setg(errp, "Object with id '%s' is not TLS credentials",

                       sock->tls_creds);

            goto error;

        }

        object_ref(OBJECT(s->tls_creds));

        if (is_listen) {

            if (s->tls_creds->endpoint != QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {

                error_setg(errp, "%s",

                           "Expected TLS credentials for server endpoint");

                goto error;

            }

        } else {

            if (s->tls_creds->endpoint != QCRYPTO_TLS_CREDS_ENDPOINT_CLIENT) {

                error_setg(errp, "%s",

                           "Expected TLS credentials for client endpoint");

                goto error;

            }

        }

    }



    s->addr = QAPI_CLONE(SocketAddress, sock->addr);



    qemu_chr_set_feature(chr, QEMU_CHAR_FEATURE_RECONNECTABLE);

    /* TODO SOCKET_ADDRESS_FD where fd has AF_UNIX */

    if (addr->type == SOCKET_ADDRESS_KIND_UNIX) {

        qemu_chr_set_feature(chr, QEMU_CHAR_FEATURE_FD_PASS);

    }



    /* be isn't opened until we get a connection */

    *be_opened = false;



    update_disconnected_filename(s);



    if (is_listen) {

        if (is_telnet || is_tn3270) {

            s->do_telnetopt = 1;

        }

    } else if (reconnect > 0) {

        s->reconnect_time = reconnect;

    }



    if (s->reconnect_time) {

        sioc = qio_channel_socket_new();

        tcp_chr_set_client_ioc_name(chr, sioc);

        qio_channel_socket_connect_async(sioc, s->addr,

                                         qemu_chr_socket_connected,

                                         chr, NULL);

    } else {

        if (s->is_listen) {

            char *name;

            sioc = qio_channel_socket_new();



            name = g_strdup_printf("chardev-tcp-listener-%s", chr->label);

            qio_channel_set_name(QIO_CHANNEL(sioc), name);

            g_free(name);



            if (qio_channel_socket_listen_sync(sioc, s->addr, errp) < 0) {

                goto error;

            }



            qapi_free_SocketAddress(s->addr);

            s->addr = socket_local_address(sioc->fd, errp);

            update_disconnected_filename(s);



            s->listen_ioc = sioc;

            if (is_waitconnect &&

                qemu_chr_wait_connected(chr, errp) < 0) {

                return;

            }

            if (!s->ioc) {

                s->listen_tag = qio_channel_add_watch(

                    QIO_CHANNEL(s->listen_ioc), G_IO_IN,

                    tcp_chr_accept, chr, NULL);

            }

        } else if (qemu_chr_wait_connected(chr, errp) < 0) {

            goto error;

        }

    }



    return;



error:

    if (sioc) {

        object_unref(OBJECT(sioc));

    }

}
