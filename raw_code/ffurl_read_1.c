int ffurl_read(URLContext *h, unsigned char *buf, int size)

{

    if (h->flags & AVIO_FLAG_WRITE)

        return AVERROR(EIO);

    return retry_transfer_wrapper(h, buf, size, 1, h->prot->url_read);

}
