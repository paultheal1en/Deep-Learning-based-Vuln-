static int nbd_negotiate_options(NBDClient *client)

{

    uint32_t flags;

    bool fixedNewstyle = false;



    /* Client sends:

        [ 0 ..   3]   client flags



        [ 0 ..   7]   NBD_OPTS_MAGIC

        [ 8 ..  11]   NBD option

        [12 ..  15]   Data length

        ...           Rest of request



        [ 0 ..   7]   NBD_OPTS_MAGIC

        [ 8 ..  11]   Second NBD option

        [12 ..  15]   Data length

        ...           Rest of request

    */



    if (nbd_negotiate_read(client->ioc, &flags, sizeof(flags)) !=

        sizeof(flags)) {

        LOG("read failed");

        return -EIO;

    }

    TRACE("Checking client flags");

    be32_to_cpus(&flags);

    if (flags & NBD_FLAG_C_FIXED_NEWSTYLE) {

        TRACE("Client supports fixed newstyle handshake");

        fixedNewstyle = true;

        flags &= ~NBD_FLAG_C_FIXED_NEWSTYLE;

    }

    if (flags != 0) {

        TRACE("Unknown client flags 0x%" PRIx32 " received", flags);

        return -EIO;

    }



    while (1) {

        int ret;

        uint32_t clientflags, length;

        uint64_t magic;



        if (nbd_negotiate_read(client->ioc, &magic, sizeof(magic)) !=

            sizeof(magic)) {

            LOG("read failed");

            return -EINVAL;

        }

        TRACE("Checking opts magic");

        if (magic != be64_to_cpu(NBD_OPTS_MAGIC)) {

            LOG("Bad magic received");

            return -EINVAL;

        }



        if (nbd_negotiate_read(client->ioc, &clientflags,

                               sizeof(clientflags)) != sizeof(clientflags)) {

            LOG("read failed");

            return -EINVAL;

        }

        clientflags = be32_to_cpu(clientflags);



        if (nbd_negotiate_read(client->ioc, &length, sizeof(length)) !=

            sizeof(length)) {

            LOG("read failed");

            return -EINVAL;

        }

        length = be32_to_cpu(length);



        TRACE("Checking option 0x%" PRIx32, clientflags);

        if (client->tlscreds &&

            client->ioc == (QIOChannel *)client->sioc) {

            QIOChannel *tioc;

            if (!fixedNewstyle) {

                TRACE("Unsupported option 0x%" PRIx32, clientflags);

                return -EINVAL;

            }

            switch (clientflags) {

            case NBD_OPT_STARTTLS:

                tioc = nbd_negotiate_handle_starttls(client, length);

                if (!tioc) {

                    return -EIO;

                }

                object_unref(OBJECT(client->ioc));

                client->ioc = QIO_CHANNEL(tioc);

                break;



            case NBD_OPT_EXPORT_NAME:

                /* No way to return an error to client, so drop connection */

                TRACE("Option 0x%x not permitted before TLS", clientflags);

                return -EINVAL;



            default:

                TRACE("Option 0x%" PRIx32 " not permitted before TLS",

                      clientflags);

                if (nbd_negotiate_drop_sync(client->ioc, length) != length) {

                    return -EIO;

                }

                nbd_negotiate_send_rep(client->ioc, NBD_REP_ERR_TLS_REQD,

                                       clientflags);

                break;

            }

        } else if (fixedNewstyle) {

            switch (clientflags) {

            case NBD_OPT_LIST:

                ret = nbd_negotiate_handle_list(client, length);

                if (ret < 0) {

                    return ret;

                }

                break;



            case NBD_OPT_ABORT:

                return -EINVAL;



            case NBD_OPT_EXPORT_NAME:

                return nbd_negotiate_handle_export_name(client, length);



            case NBD_OPT_STARTTLS:

                if (nbd_negotiate_drop_sync(client->ioc, length) != length) {

                    return -EIO;

                }

                if (client->tlscreds) {

                    TRACE("TLS already enabled");

                    nbd_negotiate_send_rep(client->ioc, NBD_REP_ERR_INVALID,

                                           clientflags);

                } else {

                    TRACE("TLS not configured");

                    nbd_negotiate_send_rep(client->ioc, NBD_REP_ERR_POLICY,

                                           clientflags);

                }

                break;

            default:

                TRACE("Unsupported option 0x%" PRIx32, clientflags);

                if (nbd_negotiate_drop_sync(client->ioc, length) != length) {

                    return -EIO;

                }

                nbd_negotiate_send_rep(client->ioc, NBD_REP_ERR_UNSUP,

                                       clientflags);

                break;

            }

        } else {

            /*

             * If broken new-style we should drop the connection

             * for anything except NBD_OPT_EXPORT_NAME

             */

            switch (clientflags) {

            case NBD_OPT_EXPORT_NAME:

                return nbd_negotiate_handle_export_name(client, length);



            default:

                TRACE("Unsupported option 0x%" PRIx32, clientflags);

                return -EINVAL;

            }

        }

    }

}
