static void nfs_client_close(NFSClient *client)

{

    if (client->context) {

        if (client->fh) {

            nfs_close(client->context, client->fh);

        }

        aio_set_fd_handler(client->aio_context, nfs_get_fd(client->context),

                           false, NULL, NULL, NULL, NULL);

        nfs_destroy_context(client->context);

    }

    memset(client, 0, sizeof(NFSClient));

}
