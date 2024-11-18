static int vnc_display_get_address(const char *addrstr,

                                   bool websocket,

                                   bool reverse,

                                   int displaynum,

                                   int to,

                                   bool has_ipv4,

                                   bool has_ipv6,

                                   bool ipv4,

                                   bool ipv6,

                                   SocketAddressLegacy **retaddr,

                                   Error **errp)

{

    int ret = -1;

    SocketAddressLegacy *addr = NULL;



    addr = g_new0(SocketAddressLegacy, 1);



    if (strncmp(addrstr, "unix:", 5) == 0) {

        addr->type = SOCKET_ADDRESS_LEGACY_KIND_UNIX;

        addr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

        addr->u.q_unix.data->path = g_strdup(addrstr + 5);



        if (websocket) {

            error_setg(errp, "UNIX sockets not supported with websock");

            goto cleanup;

        }



        if (to) {

            error_setg(errp, "Port range not support with UNIX socket");

            goto cleanup;

        }

        ret = 0;

    } else {

        const char *port;

        size_t hostlen;

        unsigned long long baseport = 0;

        InetSocketAddress *inet;



        port = strrchr(addrstr, ':');

        if (!port) {

            if (websocket) {

                hostlen = 0;

                port = addrstr;

            } else {

                error_setg(errp, "no vnc port specified");

                goto cleanup;

            }

        } else {

            hostlen = port - addrstr;

            port++;

            if (*port == '\0') {

                error_setg(errp, "vnc port cannot be empty");

                goto cleanup;

            }

        }



        addr->type = SOCKET_ADDRESS_LEGACY_KIND_INET;

        inet = addr->u.inet.data = g_new0(InetSocketAddress, 1);

        if (addrstr[0] == '[' && addrstr[hostlen - 1] == ']') {

            inet->host = g_strndup(addrstr + 1, hostlen - 2);

        } else {

            inet->host = g_strndup(addrstr, hostlen);

        }

        /* plain VNC port is just an offset, for websocket

         * port is absolute */

        if (websocket) {

            if (g_str_equal(addrstr, "") ||

                g_str_equal(addrstr, "on")) {

                if (displaynum == -1) {

                    error_setg(errp, "explicit websocket port is required");

                    goto cleanup;

                }

                inet->port = g_strdup_printf(

                    "%d", displaynum + 5700);

                if (to) {

                    inet->has_to = true;

                    inet->to = to + 5700;

                }

            } else {

                inet->port = g_strdup(port);

            }

        } else {

            int offset = reverse ? 0 : 5900;

            if (parse_uint_full(port, &baseport, 10) < 0) {

                error_setg(errp, "can't convert to a number: %s", port);

                goto cleanup;

            }

            if (baseport > 65535 ||

                baseport + offset > 65535) {

                error_setg(errp, "port %s out of range", port);

                goto cleanup;

            }

            inet->port = g_strdup_printf(

                "%d", (int)baseport + offset);



            if (to) {

                inet->has_to = true;

                inet->to = to + offset;

            }

        }



        inet->ipv4 = ipv4;

        inet->has_ipv4 = has_ipv4;

        inet->ipv6 = ipv6;

        inet->has_ipv6 = has_ipv6;



        ret = baseport;

    }



    *retaddr = addr;



 cleanup:

    if (ret < 0) {

        qapi_free_SocketAddressLegacy(addr);

    }

    return ret;

}
