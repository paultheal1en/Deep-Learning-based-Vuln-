static int qio_dns_resolver_lookup_sync_inet(QIODNSResolver *resolver,

                                             SocketAddress *addr,

                                             size_t *naddrs,

                                             SocketAddress ***addrs,

                                             Error **errp)

{

    struct addrinfo ai, *res, *e;

    InetSocketAddress *iaddr = addr->u.inet.data;

    char port[33];

    char uaddr[INET6_ADDRSTRLEN + 1];

    char uport[33];

    int rc;

    Error *err = NULL;

    size_t i;



    *naddrs = 0;

    *addrs = NULL;



    memset(&ai, 0, sizeof(ai));

    ai.ai_flags = AI_PASSIVE;

    if (iaddr->has_numeric && iaddr->numeric) {

        ai.ai_flags |= AI_NUMERICHOST | AI_NUMERICSERV;

    }

    ai.ai_family = inet_ai_family_from_address(iaddr, &err);

    ai.ai_socktype = SOCK_STREAM;



    if (err) {

        error_propagate(errp, err);

        return -1;

    }



    if (iaddr->host == NULL) {

        error_setg(errp, "host not specified");

        return -1;

    }

    if (iaddr->port != NULL) {

        pstrcpy(port, sizeof(port), iaddr->port);

    } else {

        port[0] = '\0';

    }



    rc = getaddrinfo(strlen(iaddr->host) ? iaddr->host : NULL,

                     strlen(port) ? port : NULL, &ai, &res);

    if (rc != 0) {

        error_setg(errp, "address resolution failed for %s:%s: %s",

                   iaddr->host, port, gai_strerror(rc));

        return -1;

    }



    for (e = res; e != NULL; e = e->ai_next) {

        (*naddrs)++;

    }



    *addrs = g_new0(SocketAddress *, *naddrs);



    /* create socket + bind */

    for (i = 0, e = res; e != NULL; i++, e = e->ai_next) {

        SocketAddress *newaddr = g_new0(SocketAddress, 1);

        InetSocketAddress *newiaddr = g_new0(InetSocketAddress, 1);

        newaddr->u.inet.data = newiaddr;

        newaddr->type = SOCKET_ADDRESS_KIND_INET;



        getnameinfo((struct sockaddr *)e->ai_addr, e->ai_addrlen,

                    uaddr, INET6_ADDRSTRLEN, uport, 32,

                    NI_NUMERICHOST | NI_NUMERICSERV);



        *newiaddr = (InetSocketAddress){

            .host = g_strdup(uaddr),

            .port = g_strdup(uport),

            .has_numeric = true,

            .numeric = true,

            .has_to = iaddr->has_to,

            .to = iaddr->to,

            .has_ipv4 = false,

            .has_ipv6 = false,

        };



        (*addrs)[i] = newaddr;

    }

    freeaddrinfo(res);

    return 0;

}
