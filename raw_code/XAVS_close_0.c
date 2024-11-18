static av_cold int XAVS_close(AVCodecContext *avctx)

{

    XavsContext *x4 = avctx->priv_data;



    av_freep(&avctx->extradata);

    av_free(x4->sei);

    av_freep(&x4->pts_buffer);



    if (x4->enc)

        xavs_encoder_close(x4->enc);



    av_frame_free(&avctx->coded_frame);



    return 0;

}
