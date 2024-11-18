void nbd_client_put(NBDClient *client)

{

    if (--client->refcount == 0) {

        /* The last reference should be dropped by client->close,

         * which is called by nbd_client_close.

         */

        assert(client->closing);



        nbd_unset_handlers(client);

        close(client->sock);

        client->sock = -1;

        if (client->exp) {

            QTAILQ_REMOVE(&client->exp->clients, client, next);

            nbd_export_put(client->exp);

        }

        g_free(client);

    }

}
