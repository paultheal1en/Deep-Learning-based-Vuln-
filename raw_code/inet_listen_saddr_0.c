static int inet_listen_saddr(InetSocketAddress *saddr,

                             int port_offset,

                             bool update_addr,

                             Error **errp)

{

    struct addrinfo ai,*res,*e;

    char port[33];

    char uaddr[INET6_ADDRSTRLEN+1];

    char uport[33];

    int slisten, rc, port_min, port_max, p;

    Error *err = NULL;



    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_PASSIVE;

    if (saddr->has_numeric && saddr->numeric) {

        ai.ai_flags |= AI_NUMERICHOST | AI_NUMERICSERV;

    }

    ai.ai_family = inet_ai_family_from_address(saddr, &err);

    ai.ai_socktype = SOCK_STREAM;



    if (err) {

        error_propagate(errp, err);

        return -1;

    }



    if (saddr->host == NULL) {

        error_setg(errp, "host not specified");

        return -1;

    }

    if (saddr->port != NULL) {

        pstrcpy(port, sizeof(port), saddr->port);

    } else {

        port[0] = '\0';

    }



    /* lookup */

    if (port_offset) {

        unsigned long long baseport;

        if (strlen(port) == 0) {

            error_setg(errp, "port not specified");

            return -1;

        }

        if (parse_uint_full(port, &baseport, 10) < 0) {

            error_setg(errp, "can't convert to a number: %s", port);

            return -1;

        }

        if (baseport > 65535 ||

            baseport + port_offset > 65535) {

            error_setg(errp, "port %s out of range", port);

            return -1;

        }

        snprintf(port, sizeof(port), "%d", (int)baseport + port_offset);

    }

    rc = getaddrinfo(strlen(saddr->host) ? saddr->host : NULL,

                     strlen(port) ? port : NULL, &ai, &res);

    if (rc != 0) {

        error_setg(errp, "address resolution failed for %s:%s: %s",

                   saddr->host, port, gai_strerror(rc));

        return -1;

    }



    /* create socket + bind */

    for (e = res; e != NULL; e = e->ai_next) {

        getnameinfo((struct sockaddr*)e->ai_addr,e->ai_addrlen,

		        uaddr,INET6_ADDRSTRLEN,uport,32,

		        NI_NUMERICHOST | NI_NUMERICSERV);

        slisten = qemu_socket(e->ai_family, e->ai_socktype, e->ai_protocol);

        if (slisten < 0) {

            if (!e->ai_next) {

                error_setg_errno(errp, errno, "Failed to create socket");

            }

            continue;

        }



        socket_set_fast_reuse(slisten);



        port_min = inet_getport(e);

        port_max = saddr->has_to ? saddr->to + port_offset : port_min;

        for (p = port_min; p <= port_max; p++) {

            inet_setport(e, p);

            if (try_bind(slisten, saddr, e) >= 0) {

                goto listen;

            }

            if (p == port_max) {

                if (!e->ai_next) {

                    error_setg_errno(errp, errno, "Failed to bind socket");

                }

            }

        }

        closesocket(slisten);

    }

    freeaddrinfo(res);

    return -1;



listen:

    if (listen(slisten,1) != 0) {

        error_setg_errno(errp, errno, "Failed to listen on socket");

        closesocket(slisten);

        freeaddrinfo(res);

        return -1;

    }

    if (update_addr) {

        g_free(saddr->host);

        saddr->host = g_strdup(uaddr);

        g_free(saddr->port);

        saddr->port = g_strdup_printf("%d",

                                      inet_getport(e) - port_offset);

        saddr->has_ipv6 = saddr->ipv6 = e->ai_family == PF_INET6;

        saddr->has_ipv4 = saddr->ipv4 = e->ai_family != PF_INET6;

    }

    freeaddrinfo(res);

    return slisten;

}
