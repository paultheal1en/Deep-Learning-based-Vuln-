int ffurl_get_file_handle(URLContext *h)

{

    if (!h->prot->url_get_file_handle)

        return -1;

    return h->prot->url_get_file_handle(h);

}
