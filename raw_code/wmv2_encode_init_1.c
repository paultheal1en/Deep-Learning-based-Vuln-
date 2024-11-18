static av_cold int wmv2_encode_init(AVCodecContext *avctx){

    Wmv2Context * const w= avctx->priv_data;



    if(ff_MPV_encode_init(avctx) < 0)

        return -1;



    ff_wmv2_common_init(w);



    avctx->extradata_size= 4;

    avctx->extradata= av_mallocz(avctx->extradata_size + 10);

    encode_ext_header(w);



    return 0;

}
