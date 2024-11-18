void nbd_client_new(NBDExport *exp, int csock, void (*close_fn)(NBDClient *))

{

    NBDClient *client;

    client = g_malloc0(sizeof(NBDClient));

    client->refcount = 1;

    client->exp = exp;

    client->sock = csock;

    client->can_read = true;

    if (nbd_send_negotiate(client)) {

        shutdown(client->sock, 2);

        close_fn(client);

        return;

    }

    client->close = close_fn;

    qemu_co_mutex_init(&client->send_lock);

    nbd_set_handlers(client);



    if (exp) {

        QTAILQ_INSERT_TAIL(&exp->clients, client, next);

        nbd_export_get(exp);

    }

}
