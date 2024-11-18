int ffurl_register_protocol(URLProtocol *protocol, int size)

{

    URLProtocol **p;

    if (size < sizeof(URLProtocol)) {

        URLProtocol *temp = av_mallocz(sizeof(URLProtocol));

        memcpy(temp, protocol, size);

        protocol = temp;

    }

    p = &first_protocol;

    while (*p != NULL)

        p = &(*p)->next;

    *p             = protocol;

    protocol->next = NULL;

    return 0;

}
