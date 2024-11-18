void net_slirp_hostfwd_remove(Monitor *mon, const char *src_str)

{

    struct in_addr host_addr = { .s_addr = INADDR_ANY };

    int host_port;

    char buf[256] = "";

    const char *p = src_str;

    int is_udp = 0;

    int n;



    if (!slirp_inited) {

        monitor_printf(mon, "user mode network stack not in use\n");

        return;

    }



    if (!src_str || !src_str[0])

        goto fail_syntax;



    get_str_sep(buf, sizeof(buf), &p, ':');



    if (!strcmp(buf, "tcp") || buf[0] == '\0') {

        is_udp = 0;

    } else if (!strcmp(buf, "udp")) {

        is_udp = 1;

    } else {

        goto fail_syntax;

    }



    if (get_str_sep(buf, sizeof(buf), &p, ':') < 0) {

        goto fail_syntax;

    }

    if (buf[0] != '\0' && !inet_aton(buf, &host_addr)) {

        goto fail_syntax;

    }



    host_port = atoi(p);



    n = slirp_remove_hostfwd(is_udp, host_addr, host_port);



    monitor_printf(mon, "removed %d host forwarding rules for %s\n", n,

                   src_str);

    return;



 fail_syntax:

    monitor_printf(mon, "invalid format\n");

}
