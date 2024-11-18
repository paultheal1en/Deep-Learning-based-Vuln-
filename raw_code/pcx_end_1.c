static av_cold int pcx_end(AVCodecContext *avctx) {

    PCXContext *s = avctx->priv_data;



    if(s->picture.data[0])

        avctx->release_buffer(avctx, &s->picture);



    return 0;

}
