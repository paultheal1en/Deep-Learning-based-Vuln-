int inet_connect_opts(QemuOpts *opts)

{

    struct addrinfo ai,*res,*e;

    const char *addr;

    const char *port;

    char uaddr[INET6_ADDRSTRLEN+1];

    char uport[33];

    int sock,rc;



    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_CANONNAME | AI_ADDRCONFIG;

    ai.ai_family = PF_UNSPEC;

    ai.ai_socktype = SOCK_STREAM;



    addr = qemu_opt_get(opts, "host");

    port = qemu_opt_get(opts, "port");

    if (addr == NULL || port == NULL) {

        fprintf(stderr, "inet_connect: host and/or port not specified\n");

        return -1;

    }



    if (qemu_opt_get_bool(opts, "ipv4", 0))

        ai.ai_family = PF_INET;

    if (qemu_opt_get_bool(opts, "ipv6", 0))

        ai.ai_family = PF_INET6;



    /* lookup */

    if (0 != (rc = getaddrinfo(addr, port, &ai, &res))) {

        fprintf(stderr,"getaddrinfo(%s,%s): %s\n", addr, port,

                gai_strerror(rc));

	return -1;

    }

    if (sockets_debug)

        inet_print_addrinfo(__FUNCTION__, res);



    for (e = res; e != NULL; e = e->ai_next) {

        if (getnameinfo((struct sockaddr*)e->ai_addr,e->ai_addrlen,

                            uaddr,INET6_ADDRSTRLEN,uport,32,

                            NI_NUMERICHOST | NI_NUMERICSERV) != 0) {

            fprintf(stderr,"%s: getnameinfo: oops\n", __FUNCTION__);

            continue;

        }

        sock = socket(e->ai_family, e->ai_socktype, e->ai_protocol);

        if (sock < 0) {

            fprintf(stderr,"%s: socket(%s): %s\n", __FUNCTION__,

            inet_strfamily(e->ai_family), strerror(errno));

            continue;

        }

        setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,(void*)&on,sizeof(on));



        /* connect to peer */

        if (connect(sock,e->ai_addr,e->ai_addrlen) < 0) {

            if (sockets_debug || NULL == e->ai_next)

                fprintf(stderr, "%s: connect(%s,%s,%s,%s): %s\n", __FUNCTION__,

                        inet_strfamily(e->ai_family),

                        e->ai_canonname, uaddr, uport, strerror(errno));

            closesocket(sock);

            continue;

        }

        if (sockets_debug)

            fprintf(stderr, "%s: connect(%s,%s,%s,%s): OK\n", __FUNCTION__,

                    inet_strfamily(e->ai_family),

                    e->ai_canonname, uaddr, uport);

        freeaddrinfo(res);

        return sock;

    }

    freeaddrinfo(res);

    return -1;

}
