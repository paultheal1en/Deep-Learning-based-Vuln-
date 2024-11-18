static int nbd_co_flush(BlockDriverState *bs)

{

    BDRVNBDState *s = bs->opaque;



    return nbd_client_session_co_flush(&s->client);

}
