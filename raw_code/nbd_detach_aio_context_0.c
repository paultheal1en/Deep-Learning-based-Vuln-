static void nbd_detach_aio_context(BlockDriverState *bs)

{

    BDRVNBDState *s = bs->opaque;



    nbd_client_session_detach_aio_context(&s->client);

}
