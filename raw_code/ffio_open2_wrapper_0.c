int ffio_open2_wrapper(struct AVFormatContext *s, AVIOContext **pb, const char *url, int flags,

                       const AVIOInterruptCB *int_cb, AVDictionary **options)

{

    return avio_open2(pb, url, flags, int_cb, options);

}
