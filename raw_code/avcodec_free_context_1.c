void avcodec_free_context(AVCodecContext **pavctx)

{

    AVCodecContext *avctx = *pavctx;



    if (!avctx)

        return;



    avcodec_close(avctx);



    av_freep(&avctx->extradata);

    av_freep(&avctx->subtitle_header);






    av_freep(pavctx);

}