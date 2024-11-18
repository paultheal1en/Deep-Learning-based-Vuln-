static av_cold int xwd_encode_close(AVCodecContext *avctx)

{

    av_freep(&avctx->coded_frame);



    return 0;

}
