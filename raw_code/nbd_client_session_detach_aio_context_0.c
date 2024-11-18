void nbd_client_session_detach_aio_context(NbdClientSession *client)

{

    aio_set_fd_handler(bdrv_get_aio_context(client->bs), client->sock,

                       NULL, NULL, NULL);

}
