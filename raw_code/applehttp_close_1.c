static int applehttp_close(URLContext *h)

{

    AppleHTTPContext *s = h->priv_data;



    free_segment_list(s);

    free_variant_list(s);

    ffurl_close(s->seg_hd);

    av_free(s);

    return 0;

}
