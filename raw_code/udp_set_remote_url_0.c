int udp_set_remote_url(URLContext *h, const char *uri)

{

    UDPContext *s = h->priv_data;

    char hostname[256];

    int port;

    

    url_split(NULL, 0, hostname, sizeof(hostname), &port, NULL, 0, uri);



    /* set the destination address */

    if (resolve_host(&s->dest_addr.sin_addr, hostname) < 0)

        return AVERROR_IO;

    s->dest_addr.sin_family = AF_INET;

    s->dest_addr.sin_port = htons(port);

    return 0;

}
