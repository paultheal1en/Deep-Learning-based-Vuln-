int nbd_client_session_co_discard(NbdClientSession *client, int64_t sector_num,

    int nb_sectors)

{

    struct nbd_request request;

    struct nbd_reply reply;

    ssize_t ret;



    if (!(client->nbdflags & NBD_FLAG_SEND_TRIM)) {

        return 0;

    }

    request.type = NBD_CMD_TRIM;

    request.from = sector_num * 512;

    request.len = nb_sectors * 512;



    nbd_coroutine_start(client, &request);

    ret = nbd_co_send_request(client, &request, NULL, 0);

    if (ret < 0) {

        reply.error = -ret;

    } else {

        nbd_co_receive_reply(client, &request, &reply, NULL, 0);

    }

    nbd_coroutine_end(client, &request);

    return -reply.error;



}
