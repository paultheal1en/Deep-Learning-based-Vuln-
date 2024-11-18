static void build_udp_url(char *buf, int buf_size,

                          const char *hostname, int port,

                          int local_port, int ttl)

{

    snprintf(buf, buf_size, "udp://%s:%d", hostname, port);

    if (local_port >= 0)

        url_add_option(buf, buf_size, "localport=%d", local_port);

    if (ttl >= 0)

        url_add_option(buf, buf_size, "ttl=%d", ttl);

}
