static void vp9_superframe_close(AVBSFContext *ctx)

{

    VP9BSFContext *s = ctx->priv_data;

    int n;



    // free cached data

    for (n = 0; n < s->n_cache; n++)

        av_packet_free(&s->cache[n]);

}
