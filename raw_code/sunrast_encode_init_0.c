static av_cold int sunrast_encode_init(AVCodecContext *avctx)

{

    SUNRASTContext *s = avctx->priv_data;



    switch (avctx->coder_type) {

    case FF_CODER_TYPE_RLE:

        s->type = RT_BYTE_ENCODED;

        break;

    case FF_CODER_TYPE_RAW:

        s->type = RT_STANDARD;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "invalid coder_type\n");

        return AVERROR(EINVAL);

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    avctx->coded_frame->key_frame = 1;

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    s->maptype                    = RMT_NONE;

    s->maplength                  = 0;



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_MONOWHITE:

        s->depth = 1;

        break;

    case AV_PIX_FMT_PAL8 :

        s->maptype   = RMT_EQUAL_RGB;

        s->maplength = 3 * 256;

        /* fall-through */

    case AV_PIX_FMT_GRAY8:

        s->depth = 8;

        break;

    case AV_PIX_FMT_BGR24:

        s->depth = 24;

        break;

    default:

        return AVERROR_BUG;

    }

    s->length = avctx->height * (FFALIGN(avctx->width * s->depth, 16) >> 3);

    s->size   = 32 + s->maplength +

                s->length * (s->type == RT_BYTE_ENCODED ? 2 : 1);



    return 0;

}
