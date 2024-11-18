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

    }

    r->rpr = (avctx->extradata[1] & 7) >> 1;

    r->rpr = FFMIN(r->rpr + 1, 3);

    if(avctx->extradata_size - 8 < (r->rpr - 1) * 2){

        av_log(avctx, AV_LOG_ERROR, "Insufficient extradata - need at least %d bytes, got %d\n",

               6 + r->rpr * 2, avctx->extradata_size);

        return AVERROR(EINVAL);

    }

    r->parse_slice_header = rv30_parse_slice_header;

    r->decode_intra_types = rv30_decode_intra_types;

    r->decode_mb_info     = rv30_decode_mb_info;

    r->loop_filter        = rv30_loop_filter;

    r->luma_dc_quant_i = rv30_luma_dc_quant;

    r->luma_dc_quant_p = rv30_luma_dc_quant;

    return 0;

}
