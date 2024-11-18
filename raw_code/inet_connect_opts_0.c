int inet_connect_opts(QemuOpts *opts, Error **errp,

                      NonBlockingConnectHandler *callback, void *opaque)

{

    struct addrinfo *res, *e;

    int sock = -1;

    bool in_progress;

    ConnectState *connect_state = NULL;



    res = inet_parse_connect_opts(opts, errp);

    if (!res) {

        return -1;

    }



    if (callback != NULL) {

        connect_state = g_malloc0(sizeof(*connect_state));

        connect_state->addr_list = res;

        connect_state->callback = callback;

        connect_state->opaque = opaque;

    }



    for (e = res; e != NULL; e = e->ai_next) {

        if (error_is_set(errp)) {

            error_free(*errp);

            *errp = NULL;

        }

        if (connect_state != NULL) {

            connect_state->current_addr = e;

        }

        sock = inet_connect_addr(e, &in_progress, connect_state, errp);

        if (in_progress) {

            return sock;

        } else if (sock >= 0) {

            /* non blocking socket immediate success, call callback */

            if (callback != NULL) {

                callback(sock, opaque);

            }

            break;

        }

    }

    g_free(connect_state);

    freeaddrinfo(res);

    return sock;

}
