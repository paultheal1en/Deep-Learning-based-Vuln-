static coroutine_fn int nbd_negotiate(NBDClientNewData *data)

{

    NBDClient *client = data->client;

    char buf[8 + 8 + 8 + 128];

    int rc;

    const int myflags = (NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_TRIM |

                         NBD_FLAG_SEND_FLUSH | NBD_FLAG_SEND_FUA);

    bool oldStyle;



    /* Old style negotiation header without options

        [ 0 ..   7]   passwd       ("NBDMAGIC")

        [ 8 ..  15]   magic        (NBD_CLIENT_MAGIC)

        [16 ..  23]   size

        [24 ..  25]   server flags (0)

        [26 ..  27]   export flags

        [28 .. 151]   reserved     (0)



       New style negotiation header with options

        [ 0 ..   7]   passwd       ("NBDMAGIC")

        [ 8 ..  15]   magic        (NBD_OPTS_MAGIC)

        [16 ..  17]   server flags (0)

        ....options sent....

        [18 ..  25]   size

        [26 ..  27]   export flags

        [28 .. 151]   reserved     (0)

     */



    qio_channel_set_blocking(client->ioc, false, NULL);

    rc = -EINVAL;



    TRACE("Beginning negotiation.");

    memset(buf, 0, sizeof(buf));

    memcpy(buf, "NBDMAGIC", 8);



    oldStyle = client->exp != NULL && !client->tlscreds;

    if (oldStyle) {

        assert ((client->exp->nbdflags & ~65535) == 0);

        TRACE("advertising size %" PRIu64 " and flags %x",

              client->exp->size, client->exp->nbdflags | myflags);

        stq_be_p(buf + 8, NBD_CLIENT_MAGIC);

        stq_be_p(buf + 16, client->exp->size);

        stw_be_p(buf + 26, client->exp->nbdflags | myflags);

    } else {

        stq_be_p(buf + 8, NBD_OPTS_MAGIC);

        stw_be_p(buf + 16, NBD_FLAG_FIXED_NEWSTYLE);

    }



    if (oldStyle) {

        if (client->tlscreds) {

            TRACE("TLS cannot be enabled with oldstyle protocol");

            goto fail;

        }

        if (nbd_negotiate_write(client->ioc, buf, sizeof(buf)) != sizeof(buf)) {

            LOG("write failed");

            goto fail;

        }

    } else {

        if (nbd_negotiate_write(client->ioc, buf, 18) != 18) {

            LOG("write failed");

            goto fail;

        }

        rc = nbd_negotiate_options(client);

        if (rc != 0) {

            LOG("option negotiation failed");

            goto fail;

        }



        assert ((client->exp->nbdflags & ~65535) == 0);

        TRACE("advertising size %" PRIu64 " and flags %x",

              client->exp->size, client->exp->nbdflags | myflags);

        stq_be_p(buf + 18, client->exp->size);

        stw_be_p(buf + 26, client->exp->nbdflags | myflags);

        if (nbd_negotiate_write(client->ioc, buf + 18, sizeof(buf) - 18) !=

            sizeof(buf) - 18) {

            LOG("write failed");

            goto fail;

        }

    }



    TRACE("Negotiation succeeded.");

    rc = 0;

fail:

    return rc;

}
