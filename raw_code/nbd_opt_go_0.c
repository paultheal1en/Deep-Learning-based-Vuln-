static int nbd_opt_go(QIOChannel *ioc, const char *wantname,

                      NBDExportInfo *info, Error **errp)

{

    nbd_opt_reply reply;

    uint32_t len = strlen(wantname);

    uint16_t type;

    int error;

    char *buf;



    /* The protocol requires that the server send NBD_INFO_EXPORT with

     * a non-zero flags (at least NBD_FLAG_HAS_FLAGS must be set); so

     * flags still 0 is a witness of a broken server. */

    info->flags = 0;



    trace_nbd_opt_go_start(wantname);

    buf = g_malloc(4 + len + 2 + 1);

    stl_be_p(buf, len);

    memcpy(buf + 4, wantname, len);

    /* No requests, live with whatever server sends */

    stw_be_p(buf + 4 + len, 0);

    if (nbd_send_option_request(ioc, NBD_OPT_GO, len + 6, buf, errp) < 0) {

        return -1;

    }



    while (1) {

        if (nbd_receive_option_reply(ioc, NBD_OPT_GO, &reply, errp) < 0) {

            return -1;

        }

        error = nbd_handle_reply_err(ioc, &reply, errp);

        if (error <= 0) {

            return error;

        }

        len = reply.length;



        if (reply.type == NBD_REP_ACK) {

            /* Server is done sending info and moved into transmission

               phase, but make sure it sent flags */

            if (len) {

                error_setg(errp, "server sent invalid NBD_REP_ACK");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (!info->flags) {

                error_setg(errp, "broken server omitted NBD_INFO_EXPORT");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            trace_nbd_opt_go_success();

            return 1;

        }

        if (reply.type != NBD_REP_INFO) {

            error_setg(errp, "unexpected reply type %" PRIx32 ", expected %x",

                       reply.type, NBD_REP_INFO);

            nbd_send_opt_abort(ioc);

            return -1;

        }

        if (len < sizeof(type)) {

            error_setg(errp, "NBD_REP_INFO length %" PRIu32 " is too short",

                       len);

            nbd_send_opt_abort(ioc);

            return -1;

        }

        if (nbd_read(ioc, &type, sizeof(type), errp) < 0) {

            error_prepend(errp, "failed to read info type");

            nbd_send_opt_abort(ioc);

            return -1;

        }

        len -= sizeof(type);

        be16_to_cpus(&type);

        switch (type) {

        case NBD_INFO_EXPORT:

            if (len != sizeof(info->size) + sizeof(info->flags)) {

                error_setg(errp, "remaining export info len %" PRIu32

                           " is unexpected size", len);

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (nbd_read(ioc, &info->size, sizeof(info->size), errp) < 0) {

                error_prepend(errp, "failed to read info size");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be64_to_cpus(&info->size);

            if (nbd_read(ioc, &info->flags, sizeof(info->flags), errp) < 0) {

                error_prepend(errp, "failed to read info flags");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be16_to_cpus(&info->flags);

            trace_nbd_receive_negotiate_size_flags(info->size, info->flags);

            break;



        default:

            trace_nbd_opt_go_info_unknown(type, nbd_info_lookup(type));

            if (nbd_drop(ioc, len, errp) < 0) {

                error_prepend(errp, "Failed to read info payload");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            break;

        }

    }

}
