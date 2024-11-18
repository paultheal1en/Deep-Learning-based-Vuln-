static int nbd_open(BlockDriverState *bs, QDict *options, int flags,

                    Error **errp)

{

    BDRVNBDState *s = bs->opaque;

    char *export = NULL;

    int result, sock;



    /* Pop the config into our state object. Exit if invalid. */

    result = nbd_config(s, options, &export);

    if (result != 0) {

        return result;

    }



    /* establish TCP connection, return error if it fails

     * TODO: Configurable retry-until-timeout behaviour.

     */

    sock = nbd_establish_connection(bs);

    if (sock < 0) {

        return sock;

    }



    /* NBD handshake */

    result = nbd_client_session_init(&s->client, bs, sock, export);

    g_free(export);

    return result;

}
