av_cold int ff_ivi_decode_close(AVCodecContext *avctx)
{
    IVI45DecContext *ctx = avctx->priv_data;
    ivi_free_buffers(&ctx->planes[0]);
    if (ctx->mb_vlc.cust_tab.table)
        ff_free_vlc(&ctx->mb_vlc.cust_tab);
    av_frame_free(&ctx->p_frame);
    return 0;
}