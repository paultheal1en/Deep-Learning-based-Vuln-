static int udp_socket_create(UDPContext *s, struct sockaddr_storage *addr,

                             socklen_t *addr_len, const char *localaddr)

{

    int udp_fd = -1;

    struct addrinfo *res0 = NULL, *res = NULL;

    int family = AF_UNSPEC;



    if (((struct sockaddr *) &s->dest_addr)->sa_family)

        family = ((struct sockaddr *) &s->dest_addr)->sa_family;

    res0 = udp_resolve_host(localaddr[0] ? localaddr : NULL, s->local_port,

                            SOCK_DGRAM, family, AI_PASSIVE);

    if (res0 == 0)

        goto fail;

    for (res = res0; res; res=res->ai_next) {

        udp_fd = ff_socket(res->ai_family, SOCK_DGRAM, 0);

        if (udp_fd != -1) break;

        log_net_error(NULL, AV_LOG_ERROR, "socket");

    }



    if (udp_fd < 0)

        goto fail;



    memcpy(addr, res->ai_addr, res->ai_addrlen);

    *addr_len = res->ai_addrlen;



    freeaddrinfo(res0);



    return udp_fd;



 fail:

    if (udp_fd >= 0)

        closesocket(udp_fd);

    if(res0)

        freeaddrinfo(res0);

    return -1;

}
