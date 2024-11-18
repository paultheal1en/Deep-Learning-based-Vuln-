static av_cold int roq_decode_init(AVCodecContext *avctx)

{

    RoqContext *s = avctx->priv_data;



    s->avctx = avctx;



    if (avctx->width % 16 || avctx->height % 16) {

        av_log(avctx, AV_LOG_ERROR,

               "Dimensions must be a multiple of 16\n");

        return AVERROR_PATCHWELCOME;

    }



    s->width = avctx->width;

    s->height = avctx->height;



    s->last_frame    = av_frame_alloc();

    s->current_frame = av_frame_alloc();

    if (!s->current_frame || !s->last_frame) {

        av_frame_free(&s->current_frame);

        av_frame_free(&s->last_frame);

        return AVERROR(ENOMEM);

    }



    avctx->pix_fmt = AV_PIX_FMT_YUV444P;



    return 0;

}
