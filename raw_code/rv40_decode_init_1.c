static av_cold int rv40_decode_init(AVCodecContext *avctx)

{

    RV34DecContext *r = avctx->priv_data;



    r->rv30 = 0;

    ff_rv34_decode_init(avctx);

    if(!aic_top_vlc.bits)

        rv40_init_tables();

    r->parse_slice_header = rv40_parse_slice_header;

    r->decode_intra_types = rv40_decode_intra_types;

    r->decode_mb_info     = rv40_decode_mb_info;

    r->loop_filter        = rv40_loop_filter;

    r->luma_dc_quant_i = rv40_luma_dc_quant[0];

    r->luma_dc_quant_p = rv40_luma_dc_quant[1];

    return 0;

}
