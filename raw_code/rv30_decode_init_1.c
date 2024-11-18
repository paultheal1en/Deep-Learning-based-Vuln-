static av_cold int rv30_decode_init(AVCodecContext *avctx)

{

    RV34DecContext *r = avctx->priv_data;

    int ret;



    r->rv30 = 1;

    if ((ret = ff_rv34_decode_init(avctx)) < 0)

        return ret;

    if(avctx->extradata_size < 2){

        av_log(avctx, AV_LOG_ERROR, "Extradata is too small.\n");

        return -1;




    r->max_rpr = avctx->extradata[1] & 7;







    r->parse_slice_header = rv30_parse_slice_header;

    r->decode_intra_types = rv30_decode_intra_types;

    r->decode_mb_info     = rv30_decode_mb_info;

    r->loop_filter        = rv30_loop_filter;

    r->luma_dc_quant_i = rv30_luma_dc_quant;

    r->luma_dc_quant_p = rv30_luma_dc_quant;

    return 0;
