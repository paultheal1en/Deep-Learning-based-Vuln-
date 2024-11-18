static av_cold int wmv2_decode_init(AVCodecContext *avctx){

    Wmv2Context * const w= avctx->priv_data;



    if(avctx->idct_algo==FF_IDCT_AUTO){

        avctx->idct_algo=FF_IDCT_WMV2;

    }



    if(ff_msmpeg4_decode_init(avctx) < 0)

        return -1;



    ff_wmv2_common_init(w);



    ff_intrax8_common_init(&w->x8,&w->s);



    return 0;

}
