static int xan_decode_end(AVCodecContext *avctx)

{

    XanContext *s = avctx->priv_data;



    /* release the last frame */

    avctx->release_buffer(avctx, &s->last_frame);



    av_free(s->buffer1);

    av_free(s->buffer2);



    return 0;

}
