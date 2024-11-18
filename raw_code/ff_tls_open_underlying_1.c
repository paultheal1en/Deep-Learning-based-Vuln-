int ff_tls_open_underlying(TLSShared *c, URLContext *parent, const char *uri, AVDictionary **options)

{

    int port;

    const char *p;

    char buf[200], opts[50] = "";

    struct addrinfo hints = { 0 }, *ai = NULL;

    const char *proxy_path;

    int use_proxy;



    set_options(c, uri);



    if (c->listen)

        snprintf(opts, sizeof(opts), "?listen=1");



    av_url_split(NULL, 0, NULL, 0, c->host, sizeof(c->host), &port, NULL, 0, uri);



    p = strchr(uri, '?');



    if (!p) {

        p = opts;

    } else {

        if (av_find_info_tag(opts, sizeof(opts), "listen", p))

            c->listen = 1;

    }



    ff_url_join(buf, sizeof(buf), "tcp", NULL, c->host, port, "%s", p);



    hints.ai_flags = AI_NUMERICHOST;

    if (!getaddrinfo(c->host, NULL, &hints, &ai)) {

        c->numerichost = 1;

        freeaddrinfo(ai);

    }



    proxy_path = getenv("http_proxy");

    use_proxy = !ff_http_match_no_proxy(getenv("no_proxy"), c->host) &&

                proxy_path && av_strstart(proxy_path, "http://", NULL);



    if (use_proxy) {

        char proxy_host[200], proxy_auth[200], dest[200];

        int proxy_port;

        av_url_split(NULL, 0, proxy_auth, sizeof(proxy_auth),

                     proxy_host, sizeof(proxy_host), &proxy_port, NULL, 0,

                     proxy_path);

        ff_url_join(dest, sizeof(dest), NULL, NULL, c->host, port, NULL);

        ff_url_join(buf, sizeof(buf), "httpproxy", proxy_auth, proxy_host,

                    proxy_port, "/%s", dest);

    }



    return ffurl_open(&c->tcp, buf, AVIO_FLAG_READ_WRITE,

                      &parent->interrupt_callback, options);

}
