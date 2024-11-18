static int xan_decode_init(AVCodecContext *avctx)
{
    XanContext *s = avctx->priv_data;
    int i;
    s->avctx = avctx;
    if ((avctx->codec->id == CODEC_ID_XAN_WC3) && 
        (s->avctx->palctrl == NULL)) {
        av_log(avctx, AV_LOG_ERROR, " WC3 Xan video: palette expected.\n");
    }
    avctx->pix_fmt = PIX_FMT_PAL8;
    avctx->has_b_frames = 0;
    dsputil_init(&s->dsp, avctx);
    /* initialize the RGB -> YUV tables */
    for (i = 0; i < 256; i++) {
        y_r_table[i] = Y_R * i;
        y_g_table[i] = Y_G * i;
        y_b_table[i] = Y_B * i;
        u_r_table[i] = U_R * i;
        u_g_table[i] = U_G * i;
        u_b_table[i] = U_B * i;
        v_r_table[i] = V_R * i;
        v_g_table[i] = V_G * i;
        v_b_table[i] = V_B * i;
    }
    s->buffer1 = av_malloc(avctx->width * avctx->height);
    s->buffer2 = av_malloc(avctx->width * avctx->height);
    if (!s->buffer1 || !s->buffer2)
    return 0;
}