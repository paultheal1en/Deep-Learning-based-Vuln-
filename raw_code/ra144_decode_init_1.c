static av_cold int ra144_decode_init(AVCodecContext * avctx)
{
    RA144Context *ractx = avctx->priv_data;
    ractx->avctx = avctx;
    ractx->lpc_coef[0] = ractx->lpc_tables[0];
    ractx->lpc_coef[1] = ractx->lpc_tables[1];
    avctx->channels       = 1;
    avctx->channel_layout = AV_CH_LAYOUT_MONO;
    avctx->sample_fmt     = AV_SAMPLE_FMT_S16;
    return 0;
}