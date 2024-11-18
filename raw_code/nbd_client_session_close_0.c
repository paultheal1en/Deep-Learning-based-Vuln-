void nbd_client_session_close(NbdClientSession *client)

{

    struct nbd_request request = {

        .type = NBD_CMD_DISC,

        .from = 0,

        .len = 0

    };



    if (!client->bs) {

        return;

    }

    if (client->sock == -1) {

        return;

    }



    nbd_send_request(client->sock, &request);



    nbd_teardown_connection(client);

    client->bs = NULL;

}
