static char *addr_to_string(const char *format,

                            struct sockaddr_storage *sa,

                            socklen_t salen) {

    char *addr;

    char host[NI_MAXHOST];

    char serv[NI_MAXSERV];

    int err;



    if ((err = getnameinfo((struct sockaddr *)sa, salen,

                           host, sizeof(host),

                           serv, sizeof(serv),

                           NI_NUMERICHOST | NI_NUMERICSERV)) != 0) {

        VNC_DEBUG("Cannot resolve address %d: %s\n",

                  err, gai_strerror(err));

        return NULL;

    }



    if (asprintf(&addr, format, host, serv) < 0)

        return NULL;



    return addr;

}
