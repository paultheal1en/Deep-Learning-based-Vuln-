static int nbd_open(BlockDriverState *bs, QDict *options, int flags,

                    Error **errp)

{

    BDRVNBDState *s = bs->opaque;

    char *export = NULL;

    int result, sock;

    Error *local_err = NULL;



    /* Pop the config into our state object. Exit if invalid. */

    nbd_config(s, options, &export, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return -EINVAL;

    }



    /* establish TCP connection, return error if it fails

     * TODO: Configurable retry-until-timeout behaviour.

     */

    sock = nbd_establish_connection(bs, errp);

    if (sock < 0) {


        return sock;

    }



    /* NBD handshake */

    result = nbd_client_init(bs, sock, export, errp);


    return result;

}