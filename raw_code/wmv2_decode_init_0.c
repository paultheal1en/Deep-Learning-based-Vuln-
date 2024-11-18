static av_cold int wmv2_decode_init(AVCodecContext *avctx)

{

    Wmv2Context *const w = avctx->priv_data;

    int ret;



    if ((ret = ff_msmpeg4_decode_init(avctx)) < 0)

        return ret;



    ff_wmv2_common_init(w);



    return ff_intrax8_common_init(&w->x8, &w->s.idsp, &w->s);

}
