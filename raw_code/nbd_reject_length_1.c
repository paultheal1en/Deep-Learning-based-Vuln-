static int nbd_reject_length(NBDClient *client, bool fatal, Error **errp)

{

    int ret;



    assert(client->optlen);

    if (nbd_drop(client->ioc, client->optlen, errp) < 0) {

        return -EIO;

    }

    ret = nbd_negotiate_send_rep_err(client, NBD_REP_ERR_INVALID, errp,

                                     "option '%s' should have zero length",

                                     nbd_opt_lookup(client->opt));

    if (fatal && !ret) {

        error_setg(errp, "option '%s' should have zero length",

                   nbd_opt_lookup(client->opt));

        return -EINVAL;

    }

    return ret;

}
