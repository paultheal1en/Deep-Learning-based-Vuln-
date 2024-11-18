void nbd_client_session_close(NbdClientSession *client)

{

    if (!client->bs) {

        return;

    }



    nbd_teardown_connection(client);

    client->bs = NULL;

}
