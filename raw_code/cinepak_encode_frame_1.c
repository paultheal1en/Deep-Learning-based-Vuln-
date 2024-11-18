static int cinepak_encode_frame(AVCodecContext *avctx, unsigned char *buf, int buf_size, void *data)

{

    CinepakEncContext *s = avctx->priv_data;

    AVFrame *frame = data;

    int ret;



    s->lambda = frame->quality ? frame->quality - 1 : 2 * FF_LAMBDA_SCALE;



    frame->key_frame = s->curframe == 0;

    frame->pict_type = frame->key_frame ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;



    ret = rd_frame(s, frame, buf, buf_size);



    FFSWAP(AVFrame, s->last_frame, s->best_frame);



    if (++s->curframe >= s->keyint)

        s->curframe = 0;



    return ret;

}
