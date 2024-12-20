void nbd_client_close(NBDClient *client)

{

    if (client->closing) {

        return;

    }



    client->closing = true;



    /* Force requests to finish.  They will drop their own references,

     * then we'll close the socket and free the NBDClient.

     */

    shutdown(client->sock, 2);



    /* Also tell the client, so that they release their reference.  */

    if (client->close) {

        client->close(client);

    }

}
