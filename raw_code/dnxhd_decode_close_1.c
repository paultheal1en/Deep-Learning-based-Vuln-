static av_cold int dnxhd_decode_close(AVCodecContext *avctx)

{

    DNXHDContext *ctx = avctx->priv_data;



    ff_free_vlc(&ctx->ac_vlc);

    ff_free_vlc(&ctx->dc_vlc);

    ff_free_vlc(&ctx->run_vlc);



    av_freep(&ctx->mb_scan_index);

    av_freep(&ctx->rows);



    return 0;

}
