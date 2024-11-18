static av_cold int g722_encode_close(AVCodecContext *avctx)

{

    G722Context *c = avctx->priv_data;

    int i;

    for (i = 0; i < 2; i++) {

        av_freep(&c->paths[i]);

        av_freep(&c->node_buf[i]);

        av_freep(&c->nodep_buf[i]);

    }

    return 0;

}
