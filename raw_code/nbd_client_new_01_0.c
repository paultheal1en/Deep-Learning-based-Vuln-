NBDClient *nbd_client_new(NBDExport *exp, int csock,

                          void (*close)(NBDClient *))

{

    NBDClient *client;

    client = g_malloc0(sizeof(NBDClient));

    client->refcount = 1;

    client->exp = exp;

    client->sock = csock;

    if (nbd_send_negotiate(client)) {

        g_free(client);

        return NULL;

    }

    client->close = close;

    qemu_co_mutex_init(&client->send_lock);

    qemu_set_fd_handler2(csock, nbd_can_read, nbd_read, NULL, client);



    if (exp) {

        QTAILQ_INSERT_TAIL(&exp->clients, client, next);

        nbd_export_get(exp);

    }

    return client;

}
