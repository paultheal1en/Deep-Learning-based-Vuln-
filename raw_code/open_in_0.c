static int open_in(HLSContext *c, AVIOContext **in, const char *url)

{

    AVDictionary *tmp = NULL;

    int ret;



    av_dict_copy(&tmp, c->avio_opts, 0);



    ret = avio_open2(in, url, AVIO_FLAG_READ, c->interrupt_callback, &tmp);



    av_dict_free(&tmp);

    return ret;

}
