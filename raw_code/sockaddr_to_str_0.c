static int sockaddr_to_str(char *dest, int max_len,

                           struct sockaddr_storage *ss, socklen_t ss_len,

                           struct sockaddr_storage *ps, socklen_t ps_len,

                           bool is_listen, bool is_telnet)

{

    char shost[NI_MAXHOST], sserv[NI_MAXSERV];

    char phost[NI_MAXHOST], pserv[NI_MAXSERV];

    const char *left = "", *right = "";



    switch (ss->ss_family) {

#ifndef _WIN32

    case AF_UNIX:

        return snprintf(dest, max_len, "unix:%s%s",

                        ((struct sockaddr_un *)(ss))->sun_path,

                        is_listen ? ",server" : "");

#endif

    case AF_INET6:

        left  = "[";

        right = "]";

        /* fall through */

    case AF_INET:

        getnameinfo((struct sockaddr *) ss, ss_len, shost, sizeof(shost),

                    sserv, sizeof(sserv), NI_NUMERICHOST | NI_NUMERICSERV);

        getnameinfo((struct sockaddr *) ps, ps_len, phost, sizeof(phost),

                    pserv, sizeof(pserv), NI_NUMERICHOST | NI_NUMERICSERV);

        return snprintf(dest, max_len, "%s:%s%s%s:%s%s <-> %s%s%s:%s",

                        is_telnet ? "telnet" : "tcp",

                        left, shost, right, sserv,

                        is_listen ? ",server" : "",

                        left, phost, right, pserv);



    default:

        return snprintf(dest, max_len, "unknown");

    }

}
