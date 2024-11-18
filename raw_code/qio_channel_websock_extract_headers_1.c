qio_channel_websock_extract_headers(char *buffer,

                                    QIOChannelWebsockHTTPHeader *hdrs,

                                    size_t nhdrsalloc,

                                    Error **errp)

{

    char *nl, *sep, *tmp;

    size_t nhdrs = 0;



    /*

     * First parse the HTTP protocol greeting of format:

     *

     *   $METHOD $PATH $VERSION

     *

     * e.g.

     *

     *   GET / HTTP/1.1

     */



    nl = strstr(buffer, QIO_CHANNEL_WEBSOCK_HANDSHAKE_DELIM);

    if (!nl) {

        error_setg(errp, "Missing HTTP header delimiter");

        return 0;

    }

    *nl = '\0';



    tmp = strchr(buffer, ' ');

    if (!tmp) {

        error_setg(errp, "Missing HTTP path delimiter");

        return 0;

    }

    *tmp = '\0';



    if (!g_str_equal(buffer, QIO_CHANNEL_WEBSOCK_HTTP_METHOD)) {

        error_setg(errp, "Unsupported HTTP method %s", buffer);

        return 0;

    }



    buffer = tmp + 1;

    tmp = strchr(buffer, ' ');

    if (!tmp) {

        error_setg(errp, "Missing HTTP version delimiter");

        return 0;

    }

    *tmp = '\0';



    if (!g_str_equal(buffer, QIO_CHANNEL_WEBSOCK_HTTP_PATH)) {

        error_setg(errp, "Unexpected HTTP path %s", buffer);

        return 0;

    }



    buffer = tmp + 1;



    if (!g_str_equal(buffer, QIO_CHANNEL_WEBSOCK_HTTP_VERSION)) {

        error_setg(errp, "Unsupported HTTP version %s", buffer);

        return 0;

    }



    buffer = nl + strlen(QIO_CHANNEL_WEBSOCK_HANDSHAKE_DELIM);



    /*

     * Now parse all the header fields of format

     *

     *   $NAME: $VALUE

     *

     * e.g.

     *

     *   Cache-control: no-cache

     */

    do {

        QIOChannelWebsockHTTPHeader *hdr;



        nl = strstr(buffer, QIO_CHANNEL_WEBSOCK_HANDSHAKE_DELIM);

        if (nl) {

            *nl = '\0';

        }



        sep = strchr(buffer, ':');

        if (!sep) {

            error_setg(errp, "Malformed HTTP header");

            return 0;

        }

        *sep = '\0';

        sep++;

        while (*sep == ' ') {

            sep++;

        }



        if (nhdrs >= nhdrsalloc) {

            error_setg(errp, "Too many HTTP headers");

            return 0;

        }



        hdr = &hdrs[nhdrs++];

        hdr->name = buffer;

        hdr->value = sep;



        /* Canonicalize header name for easier identification later */

        for (tmp = hdr->name; *tmp; tmp++) {

            *tmp = g_ascii_tolower(*tmp);

        }



        if (nl) {

            buffer = nl + strlen(QIO_CHANNEL_WEBSOCK_HANDSHAKE_DELIM);

        }

    } while (nl != NULL);



    return nhdrs;

}
