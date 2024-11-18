static av_cold int xbm_encode_close(AVCodecContext *avctx)

{

    av_frame_free(&avctx->coded_frame);



    return 0;

}
