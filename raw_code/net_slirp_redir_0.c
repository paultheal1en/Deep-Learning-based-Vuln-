void net_slirp_redir(const char *redir_str)

{

    int is_udp;

    char buf[256], *r;

    const char *p;

    struct in_addr guest_addr;

    int host_port, guest_port;



    if (!slirp_inited) {

        slirp_inited = 1;

        slirp_init(0, NULL);

    }



    p = redir_str;

    if (get_str_sep(buf, sizeof(buf), &p, ':') < 0)

        goto fail;

    if (!strcmp(buf, "tcp")) {

        is_udp = 0;

    } else if (!strcmp(buf, "udp")) {

        is_udp = 1;

    } else {

        goto fail;

    }



    if (get_str_sep(buf, sizeof(buf), &p, ':') < 0)

        goto fail;

    host_port = strtol(buf, &r, 0);

    if (r == buf)

        goto fail;



    if (get_str_sep(buf, sizeof(buf), &p, ':') < 0)

        goto fail;

    if (buf[0] == '\0') {

        pstrcpy(buf, sizeof(buf), "10.0.2.15");

    }

    if (!inet_aton(buf, &guest_addr))

        goto fail;



    guest_port = strtol(p, &r, 0);

    if (r == p)

        goto fail;



    if (slirp_redir(is_udp, host_port, guest_addr, guest_port) < 0) {

        fprintf(stderr, "qemu: could not set up redirection\n");

        exit(1);

    }

    return;

 fail:

    fprintf(stderr, "qemu: syntax: -redir [tcp|udp]:host-port:[guest-host]:guest-port\n");

    exit(1);

}
