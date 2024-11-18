int ffurl_alloc(URLContext **puc, const char *filename, int flags,

                const AVIOInterruptCB *int_cb)

{

    URLProtocol *up = NULL;

    char proto_str[128], proto_nested[128], *ptr;

    size_t proto_len = strspn(filename, URL_SCHEME_CHARS);



    if (filename[proto_len] != ':' || is_dos_path(filename))

        strcpy(proto_str, "file");

    else

        av_strlcpy(proto_str, filename,

                   FFMIN(proto_len + 1, sizeof(proto_str)));



    av_strlcpy(proto_nested, proto_str, sizeof(proto_nested));

    if ((ptr = strchr(proto_nested, '+')))

        *ptr = '\0';



    while (up = ffurl_protocol_next(up)) {

        if (!strcmp(proto_str, up->name))

            return url_alloc_for_protocol(puc, up, filename, flags, int_cb);

        if (up->flags & URL_PROTOCOL_FLAG_NESTED_SCHEME &&

            !strcmp(proto_nested, up->name))

            return url_alloc_for_protocol(puc, up, filename, flags, int_cb);

    }

    *puc = NULL;

    return AVERROR_PROTOCOL_NOT_FOUND;

}
