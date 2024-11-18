int nbd_client_co_pwritev(BlockDriverState *bs, uint64_t offset,

                          uint64_t bytes, QEMUIOVector *qiov, int flags)

{

    NBDClientSession *client = nbd_get_client_session(bs);

    NBDRequest request = {

        .type = NBD_CMD_WRITE,

        .from = offset,

        .len = bytes,

    };




    if (flags & BDRV_REQ_FUA) {

        assert(client->info.flags & NBD_FLAG_SEND_FUA);

        request.flags |= NBD_CMD_FLAG_FUA;

    }



    assert(bytes <= NBD_MAX_BUFFER_SIZE);



    return nbd_co_request(bs, &request, qiov);

}