static void nbd_close(BlockDriverState *bs)

{

    BDRVNBDState *s = bs->opaque;



    qemu_opts_del(s->socket_opts);

    nbd_client_session_close(&s->client);

}
