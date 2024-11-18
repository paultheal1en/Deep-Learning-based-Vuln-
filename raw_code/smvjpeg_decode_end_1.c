static av_cold int smvjpeg_decode_end(AVCodecContext *avctx)

{

    SMVJpegDecodeContext *s = avctx->priv_data;

    MJpegDecodeContext *jpg = &s->jpg;

    int ret;



    jpg->picture_ptr = NULL;

    av_frame_free(&s->picture[0]);

    av_frame_free(&s->picture[1]);

    ret = avcodec_close(s->avctx);

    av_freep(&s->avctx);

    return ret;

}
