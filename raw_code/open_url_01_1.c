static int open_url(HLSContext *c, URLContext **uc, const char *url, AVDictionary *opts)

{

    AVDictionary *tmp = NULL;

    int ret;

    const char *proto_name = avio_find_protocol_name(url);

    // only http(s) & file are allowed

    if (!av_strstart(proto_name, "http", NULL) && !av_strstart(proto_name, "file", NULL))

        return AVERROR_INVALIDDATA;

    if (!strncmp(proto_name, url, strlen(proto_name)) && url[strlen(proto_name)] == ':')

        ;

    else if (strcmp(proto_name, "file") || !strcmp(url, "file,"))

        return AVERROR_INVALIDDATA;



    av_dict_copy(&tmp, c->avio_opts, 0);

    av_dict_copy(&tmp, opts, 0);



    ret = ffurl_open(uc, url, AVIO_FLAG_READ, c->interrupt_callback, &tmp);

    if( ret >= 0) {

        // update cookies on http response with setcookies.

        URLContext *u = *uc;

        update_options(&c->cookies, "cookies", u->priv_data);

        av_dict_set(&opts, "cookies", c->cookies, 0);

    }



    av_dict_free(&tmp);



    return ret;

}
