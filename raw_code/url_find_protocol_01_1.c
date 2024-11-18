static struct URLProtocol *url_find_protocol(const char *filename)

{

    URLProtocol *up = NULL;

    char proto_str[128], proto_nested[128], *ptr;

    size_t proto_len = strspn(filename, URL_SCHEME_CHARS);



    if (filename[proto_len] != ':' &&

        (filename[proto_len] != ',' || !strchr(filename + proto_len + 1, ':')) ||

        is_dos_path(filename))

        strcpy(proto_str, "file");

    else

        av_strlcpy(proto_str, filename,

                   FFMIN(proto_len + 1, sizeof(proto_str)));



    if ((ptr = strchr(proto_str, ',')))

        *ptr = '\0';

    av_strlcpy(proto_nested, proto_str, sizeof(proto_nested));

    if ((ptr = strchr(proto_nested, '+')))

        *ptr = '\0';



    while (up = ffurl_protocol_next(up)) {

        if (!strcmp(proto_str, up->name))

            break;

        if (up->flags & URL_PROTOCOL_FLAG_NESTED_SCHEME &&

            !strcmp(proto_nested, up->name))

            break;

    }



    return up;

}
