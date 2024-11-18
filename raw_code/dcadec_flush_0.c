static av_cold void dcadec_flush(AVCodecContext *avctx)

{

    DCAContext *s = avctx->priv_data;



    ff_dca_core_flush(&s->core);

    ff_dca_xll_flush(&s->xll);

    ff_dca_lbr_flush(&s->lbr);



    s->core_residual_valid = 0;

}
