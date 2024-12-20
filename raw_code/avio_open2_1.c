int avio_open2(AVIOContext **s, const char *filename, int flags,

               const AVIOInterruptCB *int_cb, AVDictionary **options)

{

    AVIOInternal *internal;

    const URLProtocol **protocols;

    URLContext *h;

    int err;



    protocols = ffurl_get_protocols(NULL, NULL);

    if (!protocols)

        return AVERROR(ENOMEM);



    err = ffurl_open(&h, filename, flags, int_cb, options, protocols);

    if (err < 0) {

        av_freep(&protocols);

        return err;

    }



    err = ffio_fdopen(s, h);

    if (err < 0) {

        ffurl_close(h);

        av_freep(&protocols);

        return err;

    }



    internal = (*s)->opaque;

    internal->protocols = protocols;



    return 0;

}
