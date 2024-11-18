static void nbd_attach_aio_context(BlockDriverState *bs,

                                   AioContext *new_context)

{

    BDRVNBDState *s = bs->opaque;



    nbd_client_session_attach_aio_context(&s->client, new_context);

}
