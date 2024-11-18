int url_open(URLContext **puc, const char *filename, int flags)

{

    URLProtocol *up;

    const char *p;

    char proto_str[128], *q;



    p = filename;

    q = proto_str;

    while (*p != '\0' && *p != ':') {

        /* protocols can only contain alphabetic chars */

        if (!isalpha(*p))

            goto file_proto;

        if ((q - proto_str) < sizeof(proto_str) - 1)

            *q++ = *p;

        p++;

    }

    /* if the protocol has length 1, we consider it is a dos drive */

    if (*p == '\0' || (q - proto_str) <= 1) {

    file_proto:

        strcpy(proto_str, "file");

    } else {

        *q = '\0';

    }



    up = first_protocol;

    while (up != NULL) {

        if (!strcmp(proto_str, up->name))

            return url_open_protocol (puc, up, filename, flags);

        up = up->next;

    }

    *puc = NULL;

    return AVERROR(ENOENT);

}
