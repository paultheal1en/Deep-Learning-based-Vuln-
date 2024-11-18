static av_cold int pcx_init(AVCodecContext *avctx) {

    PCXContext *s = avctx->priv_data;



    avcodec_get_frame_defaults(&s->picture);

    avctx->coded_frame= &s->picture;



    return 0;

}
