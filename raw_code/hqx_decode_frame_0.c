static int hqx_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_picture_ptr, AVPacket *avpkt)

{

    HQXContext *ctx = avctx->priv_data;

    uint8_t *src = avpkt->data;

    uint32_t info_tag;

    int data_start;

    int i, ret;



    if (avpkt->size < 4 + 4) {

        av_log(avctx, AV_LOG_ERROR, "Frame is too small %d.\n", avpkt->size);

        return AVERROR_INVALIDDATA;

    }



    info_tag    = AV_RL32(src);

    if (info_tag == MKTAG('I', 'N', 'F', 'O')) {

        int info_offset = AV_RL32(src + 4);

        if (info_offset > UINT32_MAX - 8 || info_offset + 8 > avpkt->size) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid INFO header offset: 0x%08"PRIX32" is too large.\n",

                   info_offset);

            return AVERROR_INVALIDDATA;

        }

        ff_canopus_parse_info_tag(avctx, src + 8, info_offset);



        info_offset += 8;

        src         += info_offset;

    }



    data_start     = src - avpkt->data;

    ctx->data_size = avpkt->size - data_start;

    ctx->src       = src;

    ctx->pic       = data;



    if (ctx->data_size < HQX_HEADER_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "Frame too small.\n");

        return AVERROR_INVALIDDATA;

    }



    if (src[0] != 'H' || src[1] != 'Q') {

        av_log(avctx, AV_LOG_ERROR, "Not an HQX frame.\n");

        return AVERROR_INVALIDDATA;

    }

    ctx->interlaced = !(src[2] & 0x80);

    ctx->format     = src[2] & 7;

    ctx->dcb        = (src[3] & 3) + 8;

    ctx->width      = AV_RB16(src + 4);

    ctx->height     = AV_RB16(src + 6);

    for (i = 0; i < 17; i++)

        ctx->slice_off[i] = AV_RB24(src + 8 + i * 3);



    if (ctx->dcb == 8) {

        av_log(avctx, AV_LOG_ERROR, "Invalid DC precision %d.\n", ctx->dcb);

        return AVERROR_INVALIDDATA;

    }

    ret = av_image_check_size(ctx->width, ctx->height, 0, avctx);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Invalid stored dimenstions %dx%d.\n",

               ctx->width, ctx->height);

        return AVERROR_INVALIDDATA;

    }



    avctx->coded_width         = FFALIGN(ctx->width,  16);

    avctx->coded_height        = FFALIGN(ctx->height, 16);

    avctx->width               = ctx->width;

    avctx->height              = ctx->height;

    avctx->bits_per_raw_sample = 10;



    switch (ctx->format) {

    case HQX_422:

        avctx->pix_fmt = AV_PIX_FMT_YUV422P16;

        ctx->decode_func = hqx_decode_422;

        break;

    case HQX_444:

        avctx->pix_fmt = AV_PIX_FMT_YUV444P16;

        ctx->decode_func = hqx_decode_444;

        break;

    case HQX_422A:

        avctx->pix_fmt = AV_PIX_FMT_YUVA422P16;

        ctx->decode_func = hqx_decode_422a;

        break;

    case HQX_444A:

        avctx->pix_fmt = AV_PIX_FMT_YUVA444P16;

        ctx->decode_func = hqx_decode_444a;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Invalid format: %d.\n", ctx->format);

        return AVERROR_INVALIDDATA;

    }



    ret = ff_get_buffer(avctx, ctx->pic, 0);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Could not allocate buffer.\n");

        return ret;

    }



    avctx->execute2(avctx, decode_slice_thread, NULL, NULL, 16);



    ctx->pic->key_frame = 1;

    ctx->pic->pict_type = AV_PICTURE_TYPE_I;



    *got_picture_ptr = 1;



    return avpkt->size;

}
