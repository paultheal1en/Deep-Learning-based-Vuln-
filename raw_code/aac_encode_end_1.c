static av_cold int aac_encode_end(AVCodecContext *avctx)

{

    AACEncContext *s = avctx->priv_data;



    ff_mdct_end(&s->mdct1024);

    ff_mdct_end(&s->mdct128);

    ff_psy_end(&s->psy);

    ff_psy_preprocess_end(s->psypp);

    av_freep(&s->samples);

    av_freep(&s->cpe);

    return 0;

}
