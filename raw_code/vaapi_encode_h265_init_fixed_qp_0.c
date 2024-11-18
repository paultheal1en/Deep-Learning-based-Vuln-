static av_cold int vaapi_encode_h265_init_fixed_qp(AVCodecContext *avctx)

{

    VAAPIEncodeContext      *ctx = avctx->priv_data;

    VAAPIEncodeH265Context *priv = ctx->priv_data;

    VAAPIEncodeH265Options  *opt = ctx->codec_options;



    priv->fixed_qp_p = opt->qp;

    if (avctx->i_quant_factor > 0.0)

        priv->fixed_qp_idr = (int)((priv->fixed_qp_p * avctx->i_quant_factor +

                                    avctx->i_quant_offset) + 0.5);

    else

        priv->fixed_qp_idr = priv->fixed_qp_p;

    if (avctx->b_quant_factor > 0.0)

        priv->fixed_qp_b = (int)((priv->fixed_qp_p * avctx->b_quant_factor +

                                  avctx->b_quant_offset) + 0.5);

    else

        priv->fixed_qp_b = priv->fixed_qp_p;



    av_log(avctx, AV_LOG_DEBUG, "Using fixed QP = "

           "%d / %d / %d for IDR- / P- / B-frames.\n",

           priv->fixed_qp_idr, priv->fixed_qp_p, priv->fixed_qp_b);

    return 0;

}
