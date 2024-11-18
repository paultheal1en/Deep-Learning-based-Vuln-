int nbd_receive_negotiate(QIOChannel *ioc, const char *name, uint32_t *flags,

                          QCryptoTLSCreds *tlscreds, const char *hostname,

                          QIOChannel **outioc,

                          off_t *size, Error **errp)

{

    char buf[256];

    uint64_t magic, s;

    int rc;



    TRACE("Receiving negotiation tlscreds=%p hostname=%s.",

          tlscreds, hostname ? hostname : "<null>");



    rc = -EINVAL;



    if (outioc) {

        *outioc = NULL;

    }

    if (tlscreds && !outioc) {

        error_setg(errp, "Output I/O channel required for TLS");

        goto fail;

    }



    if (read_sync(ioc, buf, 8) != 8) {

        error_setg(errp, "Failed to read data");

        goto fail;

    }



    buf[8] = '\0';

    if (strlen(buf) == 0) {

        error_setg(errp, "Server connection closed unexpectedly");

        goto fail;

    }



    TRACE("Magic is %c%c%c%c%c%c%c%c",

          qemu_isprint(buf[0]) ? buf[0] : '.',

          qemu_isprint(buf[1]) ? buf[1] : '.',

          qemu_isprint(buf[2]) ? buf[2] : '.',

          qemu_isprint(buf[3]) ? buf[3] : '.',

          qemu_isprint(buf[4]) ? buf[4] : '.',

          qemu_isprint(buf[5]) ? buf[5] : '.',

          qemu_isprint(buf[6]) ? buf[6] : '.',

          qemu_isprint(buf[7]) ? buf[7] : '.');



    if (memcmp(buf, "NBDMAGIC", 8) != 0) {

        error_setg(errp, "Invalid magic received");

        goto fail;

    }



    if (read_sync(ioc, &magic, sizeof(magic)) != sizeof(magic)) {

        error_setg(errp, "Failed to read magic");

        goto fail;

    }

    magic = be64_to_cpu(magic);

    TRACE("Magic is 0x%" PRIx64, magic);



    if (magic == NBD_OPTS_MAGIC) {

        uint32_t clientflags = 0;

        uint32_t opt;

        uint32_t namesize;

        uint16_t globalflags;

        uint16_t exportflags;

        bool fixedNewStyle = false;



        if (read_sync(ioc, &globalflags, sizeof(globalflags)) !=

            sizeof(globalflags)) {

            error_setg(errp, "Failed to read server flags");

            goto fail;

        }

        globalflags = be16_to_cpu(globalflags);

        *flags = globalflags << 16;

        TRACE("Global flags are %" PRIx32, globalflags);

        if (globalflags & NBD_FLAG_FIXED_NEWSTYLE) {

            fixedNewStyle = true;

            TRACE("Server supports fixed new style");

            clientflags |= NBD_FLAG_C_FIXED_NEWSTYLE;

        }

        /* client requested flags */

        clientflags = cpu_to_be32(clientflags);

        if (write_sync(ioc, &clientflags, sizeof(clientflags)) !=

            sizeof(clientflags)) {

            error_setg(errp, "Failed to send clientflags field");

            goto fail;

        }

        if (tlscreds) {

            if (fixedNewStyle) {

                *outioc = nbd_receive_starttls(ioc, tlscreds, hostname, errp);

                if (!*outioc) {

                    goto fail;

                }

                ioc = *outioc;

            } else {

                error_setg(errp, "Server does not support STARTTLS");

                goto fail;

            }

        }

        if (!name) {

            TRACE("Using default NBD export name \"\"");

            name = "";

        }

        if (fixedNewStyle) {

            /* Check our desired export is present in the

             * server export list. Since NBD_OPT_EXPORT_NAME

             * cannot return an error message, running this

             * query gives us good error reporting if the

             * server required TLS

             */

            if (nbd_receive_query_exports(ioc, name, errp) < 0) {

                goto fail;

            }

        }

        /* write the export name */

        magic = cpu_to_be64(magic);

        if (write_sync(ioc, &magic, sizeof(magic)) != sizeof(magic)) {

            error_setg(errp, "Failed to send export name magic");

            goto fail;

        }

        opt = cpu_to_be32(NBD_OPT_EXPORT_NAME);

        if (write_sync(ioc, &opt, sizeof(opt)) != sizeof(opt)) {

            error_setg(errp, "Failed to send export name option number");

            goto fail;

        }

        namesize = cpu_to_be32(strlen(name));

        if (write_sync(ioc, &namesize, sizeof(namesize)) !=

            sizeof(namesize)) {

            error_setg(errp, "Failed to send export name length");

            goto fail;

        }

        if (write_sync(ioc, (char *)name, strlen(name)) != strlen(name)) {

            error_setg(errp, "Failed to send export name");

            goto fail;

        }



        if (read_sync(ioc, &s, sizeof(s)) != sizeof(s)) {

            error_setg(errp, "Failed to read export length");

            goto fail;

        }

        *size = be64_to_cpu(s);

        TRACE("Size is %" PRIu64, *size);



        if (read_sync(ioc, &exportflags, sizeof(exportflags)) !=

            sizeof(exportflags)) {

            error_setg(errp, "Failed to read export flags");

            goto fail;

        }

        exportflags = be16_to_cpu(exportflags);

        *flags |= exportflags;

        TRACE("Export flags are %" PRIx16, exportflags);

    } else if (magic == NBD_CLIENT_MAGIC) {

        if (name) {

            error_setg(errp, "Server does not support export names");

            goto fail;

        }

        if (tlscreds) {

            error_setg(errp, "Server does not support STARTTLS");

            goto fail;

        }



        if (read_sync(ioc, &s, sizeof(s)) != sizeof(s)) {

            error_setg(errp, "Failed to read export length");

            goto fail;

        }

        *size = be64_to_cpu(s);

        TRACE("Size is %" PRIu64, *size);



        if (read_sync(ioc, flags, sizeof(*flags)) != sizeof(*flags)) {

            error_setg(errp, "Failed to read export flags");

            goto fail;

        }

        *flags = be32_to_cpu(*flags);

    } else {

        error_setg(errp, "Bad magic received");

        goto fail;

    }



    if (read_sync(ioc, &buf, 124) != 124) {

        error_setg(errp, "Failed to read reserved block");

        goto fail;

    }

    rc = 0;



fail:

    return rc;

}
