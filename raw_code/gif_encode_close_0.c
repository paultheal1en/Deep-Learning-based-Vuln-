static int gif_encode_close(AVCodecContext *avctx)

{

    GIFContext *s = avctx->priv_data;



    av_frame_free(&avctx->coded_frame);



    av_freep(&s->lzw);

    av_freep(&s->buf);

    return 0;

}
