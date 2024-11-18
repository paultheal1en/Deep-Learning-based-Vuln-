test_tls_get_ipaddr(const char *addrstr,

                    char **data,

                    int *datalen)

{

    struct addrinfo *res;

    struct addrinfo hints;



    memset(&hints, 0, sizeof(hints));

    hints.ai_flags = AI_NUMERICHOST;

    g_assert(getaddrinfo(addrstr, NULL, &hints, &res) == 0);



    *datalen = res->ai_addrlen;

    *data = g_new(char, *datalen);

    memcpy(*data, res->ai_addr, *datalen);


}