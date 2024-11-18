int avio_open2(AVIOContext **s, const char *filename, int flags,

               const AVIOInterruptCB *int_cb, AVDictionary **options)

{

    URLContext *h;

    int err;



    err = ffurl_open(&h, filename, flags, int_cb, options);

    if (err < 0)

        return err;

    err = ffio_fdopen(s, h);

    if (err < 0) {

        ffurl_close(h);

        return err;

    }

    return 0;

}
