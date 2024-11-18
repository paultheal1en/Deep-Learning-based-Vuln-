static av_cold int X264_close(AVCodecContext *avctx)

{

    X264Context *x4 = avctx->priv_data;



    av_freep(&avctx->extradata);

    av_free(x4->sei);



    if (x4->enc)

        x264_encoder_close(x4->enc);



    av_frame_free(&avctx->coded_frame);



    return 0;

}
