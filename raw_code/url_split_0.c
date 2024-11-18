void url_split(char *proto, int proto_size,

               char *hostname, int hostname_size,

               int *port_ptr,

               char *path, int path_size,

               const char *url)

{

    const char *p;

    char *q;

    int port;



    port = -1;



    p = url;

    q = proto;

    while (*p != ':' && *p != '\0') {

        if ((q - proto) < proto_size - 1)

            *q++ = *p;

        p++;

    }

    if (proto_size > 0)

        *q = '\0';

    if (*p == '\0') {

        if (proto_size > 0)

            proto[0] = '\0';

        if (hostname_size > 0)

            hostname[0] = '\0';

        p = url;

    } else {

        p++;

        if (*p == '/')

            p++;

        if (*p == '/')

            p++;

        q = hostname;

        while (*p != ':' && *p != '/' && *p != '?' && *p != '\0') {

            if ((q - hostname) < hostname_size - 1)

                *q++ = *p;

            p++;

        }

        if (hostname_size > 0)

            *q = '\0';

        if (*p == ':') {

            p++;

            port = strtoul(p, (char **)&p, 10);

        }

    }

    if (port_ptr)

        *port_ptr = port;

    pstrcpy(path, path_size, p);

}
