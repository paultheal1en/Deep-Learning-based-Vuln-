static int http_open(URLContext *h, const char *uri, int flags)

{

    HTTPContext *s = h->priv_data;



    h->is_streamed = 1;



    s->filesize = -1;

    av_strlcpy(s->location, uri, sizeof(s->location));



    if (s->headers) {

        int len = strlen(s->headers);

        if (len < 2 || strcmp("\r\n", s->headers + len - 2))

            av_log(h, AV_LOG_WARNING, "No trailing CRLF found in HTTP header.\n");

    }



    return http_open_cnx(h);

}
