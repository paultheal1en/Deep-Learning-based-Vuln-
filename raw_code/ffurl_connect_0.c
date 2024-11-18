int ffurl_connect(URLContext *uc, AVDictionary **options)

{

    int err =

        uc->prot->url_open2 ? uc->prot->url_open2(uc,

                                                  uc->filename,

                                                  uc->flags,

                                                  options) :

        uc->prot->url_open(uc, uc->filename, uc->flags);

    if (err)

        return err;

    uc->is_connected = 1;

    /* We must be careful here as ffurl_seek() could be slow,

     * for example for http */

    if ((uc->flags & AVIO_FLAG_WRITE) || !strcmp(uc->prot->name, "file"))

        if (!uc->is_streamed && ffurl_seek(uc, 0, SEEK_SET) < 0)

            uc->is_streamed = 1;

    return 0;

}
