int inet_listen_opts(QemuOpts *opts, int port_offset, Error **errp)

{

    struct addrinfo ai,*res,*e;

    const char *addr;

    char port[33];

    char uaddr[INET6_ADDRSTRLEN+1];

    char uport[33];

    int slisten, rc, to, port_min, port_max, p;



    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;

    ai.ai_family = PF_UNSPEC;

    ai.ai_socktype = SOCK_STREAM;



    if ((qemu_opt_get(opts, "host") == NULL) ||

        (qemu_opt_get(opts, "port") == NULL)) {

        error_setg(errp, "host and/or port not specified");

        return -1;

    }

    pstrcpy(port, sizeof(port), qemu_opt_get(opts, "port"));

    addr = qemu_opt_get(opts, "host");



    to = qemu_opt_get_number(opts, "to", 0);

    if (qemu_opt_get_bool(opts, "ipv4", 0))

        ai.ai_family = PF_INET;

    if (qemu_opt_get_bool(opts, "ipv6", 0))

        ai.ai_family = PF_INET6;



    /* lookup */

    if (port_offset) {

        unsigned long long baseport;

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

    rc = getaddrinfo(strlen(addr) ? addr : NULL, port, &ai, &res);

    if (rc != 0) {

        error_setg(errp, "address resolution failed for %s:%s: %s", addr, port,

                   gai_strerror(rc));

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

#ifdef IPV6_V6ONLY

        if (e->ai_family == PF_INET6) {

            /* listen on both ipv4 and ipv6 */

            const int off = 0;

            qemu_setsockopt(slisten, IPPROTO_IPV6, IPV6_V6ONLY, &off,

                            sizeof(off));

        }

#endif



        port_min = inet_getport(e);

        port_max = to ? to + port_offset : port_min;

        for (p = port_min; p <= port_max; p++) {

            inet_setport(e, p);

            if (bind(slisten, e->ai_addr, e->ai_addrlen) == 0) {

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

    qemu_opt_set(opts, "host", uaddr, &error_abort);

    qemu_opt_set_number(opts, "port", inet_getport(e) - port_offset,

                        &error_abort);

    qemu_opt_set_bool(opts, "ipv6", e->ai_family == PF_INET6,

                      &error_abort);

    qemu_opt_set_bool(opts, "ipv4", e->ai_family != PF_INET6,

                      &error_abort);

    freeaddrinfo(res);

    return slisten;

}
