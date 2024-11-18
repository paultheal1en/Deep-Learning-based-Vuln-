static int nbd_co_request(BlockDriverState *bs,

                          NBDRequest *request,

                          QEMUIOVector *qiov)

{

    NBDClientSession *client = nbd_get_client_session(bs);

    int ret;



    assert(!qiov || request->type == NBD_CMD_WRITE ||

           request->type == NBD_CMD_READ);

    ret = nbd_co_send_request(bs, request,

                              request->type == NBD_CMD_WRITE ? qiov : NULL);

    if (ret < 0) {

        return ret;

    }



    return nbd_co_receive_reply(client, request,

                                request->type == NBD_CMD_READ ? qiov : NULL);

}
