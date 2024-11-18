static int nbd_receive_list(QIOChannel *ioc, char **name, Error **errp)

{

    uint64_t magic;

    uint32_t opt;

    uint32_t type;

    uint32_t len;

    uint32_t namelen;

    int error;



    *name = NULL;

    if (read_sync(ioc, &magic, sizeof(magic)) != sizeof(magic)) {

        error_setg(errp, "failed to read list option magic");

        return -1;

    }

    magic = be64_to_cpu(magic);

    if (magic != NBD_REP_MAGIC) {

        error_setg(errp, "Unexpected option list magic");

        return -1;

    }

    if (read_sync(ioc, &opt, sizeof(opt)) != sizeof(opt)) {

        error_setg(errp, "failed to read list option");

        return -1;

    }

    opt = be32_to_cpu(opt);

    if (opt != NBD_OPT_LIST) {

        error_setg(errp, "Unexpected option type %" PRIx32 " expected %x",

                   opt, NBD_OPT_LIST);

        return -1;

    }



    if (read_sync(ioc, &type, sizeof(type)) != sizeof(type)) {

        error_setg(errp, "failed to read list option type");

        return -1;

    }

    type = be32_to_cpu(type);

    error = nbd_handle_reply_err(ioc, opt, type, errp);

    if (error <= 0) {

        return error;

    }



    if (read_sync(ioc, &len, sizeof(len)) != sizeof(len)) {

        error_setg(errp, "failed to read option length");

        return -1;

    }

    len = be32_to_cpu(len);



    if (type == NBD_REP_ACK) {

        if (len != 0) {

            error_setg(errp, "length too long for option end");

            return -1;

        }

    } else if (type == NBD_REP_SERVER) {

        if (len < sizeof(namelen) || len > NBD_MAX_BUFFER_SIZE) {

            error_setg(errp, "incorrect option length");

            return -1;

        }

        if (read_sync(ioc, &namelen, sizeof(namelen)) != sizeof(namelen)) {

            error_setg(errp, "failed to read option name length");

            return -1;

        }

        namelen = be32_to_cpu(namelen);

        len -= sizeof(namelen);

        if (len < namelen) {

            error_setg(errp, "incorrect option name length");

            return -1;

        }

        if (namelen > 255) {

            error_setg(errp, "export name length too long %" PRIu32, namelen);

            return -1;

        }



        *name = g_new0(char, namelen + 1);

        if (read_sync(ioc, *name, namelen) != namelen) {

            error_setg(errp, "failed to read export name");

            g_free(*name);

            *name = NULL;

            return -1;

        }

        (*name)[namelen] = '\0';

        len -= namelen;

        if (len) {

            char *buf = g_malloc(len + 1);

            if (read_sync(ioc, buf, len) != len) {

                error_setg(errp, "failed to read export description");

                g_free(*name);

                g_free(buf);

                *name = NULL;

                return -1;

            }

            buf[len] = '\0';

            TRACE("Ignoring export description: %s", buf);

            g_free(buf);

        }

    } else {

        error_setg(errp, "Unexpected reply type %" PRIx32 " expected %x",

                   type, NBD_REP_SERVER);

        return -1;

    }

    return 1;

}
