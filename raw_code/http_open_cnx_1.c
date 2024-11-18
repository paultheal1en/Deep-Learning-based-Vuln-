static int http_open_cnx(URLContext *h)

{

    const char *path, *proxy_path, *lower_proto = "tcp", *local_path;

    char hostname[1024], hoststr[1024], proto[10];

    char auth[1024], proxyauth[1024];

    char path1[1024];

    char buf[1024], urlbuf[1024];

    int port, use_proxy, err, location_changed = 0, redirects = 0;

    HTTPAuthType cur_auth_type, cur_proxy_auth_type;

    HTTPContext *s = h->priv_data;

    URLContext *hd = NULL;



    proxy_path = getenv("http_proxy");

    use_proxy = (proxy_path != NULL) && !getenv("no_proxy") &&

        av_strstart(proxy_path, "http://", NULL);



    /* fill the dest addr */

 redo:

    /* needed in any case to build the host string */

    av_url_split(proto, sizeof(proto), auth, sizeof(auth),

                 hostname, sizeof(hostname), &port,

                 path1, sizeof(path1), s->location);

    ff_url_join(hoststr, sizeof(hoststr), NULL, NULL, hostname, port, NULL);



    if (!strcmp(proto, "https")) {

        lower_proto = "tls";

        use_proxy = 0;

        if (port < 0)

            port = 443;

    }

    if (port < 0)

        port = 80;



    if (path1[0] == '\0')

        path = "/";

    else

        path = path1;

    local_path = path;

    if (use_proxy) {

        /* Reassemble the request URL without auth string - we don't

         * want to leak the auth to the proxy. */

        ff_url_join(urlbuf, sizeof(urlbuf), proto, NULL, hostname, port, "%s",

                    path1);

        path = urlbuf;

        av_url_split(NULL, 0, proxyauth, sizeof(proxyauth),

                     hostname, sizeof(hostname), &port, NULL, 0, proxy_path);

    }



    ff_url_join(buf, sizeof(buf), lower_proto, NULL, hostname, port, NULL);

    err = ffurl_open(&hd, buf, AVIO_FLAG_READ_WRITE,

                     &h->interrupt_callback, NULL);

    if (err < 0)

        goto fail;



    s->hd = hd;

    cur_auth_type = s->auth_state.auth_type;

    cur_proxy_auth_type = s->auth_state.auth_type;

    if (http_connect(h, path, local_path, hoststr, auth, proxyauth, &location_changed) < 0)

        goto fail;

    if (s->http_code == 401) {

        if (cur_auth_type == HTTP_AUTH_NONE && s->auth_state.auth_type != HTTP_AUTH_NONE) {

            ffurl_close(hd);

            goto redo;

        } else

            goto fail;

    }

    if (s->http_code == 407) {

        if (cur_proxy_auth_type == HTTP_AUTH_NONE &&

            s->proxy_auth_state.auth_type != HTTP_AUTH_NONE) {

            ffurl_close(hd);

            goto redo;

        } else

            goto fail;

    }

    if ((s->http_code == 301 || s->http_code == 302 || s->http_code == 303 || s->http_code == 307)

        && location_changed == 1) {

        /* url moved, get next */

        ffurl_close(hd);

        if (redirects++ >= MAX_REDIRECTS)

            return AVERROR(EIO);

        location_changed = 0;

        goto redo;

    }

    return 0;

 fail:

    if (hd)

        ffurl_close(hd);

    s->hd = NULL;

    return AVERROR(EIO);

}
