static int cdxl_decode_frame(AVCodecContext *avctx, void *data,

                             int *data_size, AVPacket *pkt)

{

    CDXLVideoContext *c = avctx->priv_data;

    AVFrame * const p = &c->frame;

    int ret, w, h, encoding, format, buf_size = pkt->size;

    const uint8_t *buf = pkt->data;



    if (buf_size < 32)

        return AVERROR_INVALIDDATA;

    encoding        = buf[1] & 7;

    format          = buf[1] & 0xE0;

    w               = AV_RB16(&buf[14]);

    h               = AV_RB16(&buf[16]);

    c->bpp          = buf[19];

    c->palette_size = AV_RB16(&buf[20]);

    c->palette      = buf + 32;

    c->video        = c->palette + c->palette_size;

    c->video_size   = buf_size - c->palette_size - 32;



    if (c->palette_size > 512)

        return AVERROR_INVALIDDATA;

    if (buf_size < c->palette_size + 32)

        return AVERROR_INVALIDDATA;

    if (c->bpp < 1)

        return AVERROR_INVALIDDATA;

    if (c->bpp > 8) {

        av_log_ask_for_sample(avctx, "unsupported pixel size: %d\n", c->bpp);

        return AVERROR_PATCHWELCOME;

    }

    if (format) {

        av_log_ask_for_sample(avctx, "unsupported pixel format: %d\n", format);

        return AVERROR_PATCHWELCOME;

    }



    if ((ret = av_image_check_size(w, h, 0, avctx)) < 0)

        return ret;

    if (w != avctx->width || h != avctx->height)

        avcodec_set_dimensions(avctx, w, h);



    if (c->video_size < FFALIGN(avctx->width, 16) * avctx->height * c->bpp / 8)

        return AVERROR_INVALIDDATA;

    if (!encoding && c->palette_size && c->bpp <= 8) {

        avctx->pix_fmt = PIX_FMT_PAL8;

    } else if (encoding == 1 && (c->bpp == 6 || c->bpp == 8)) {

        if (c->palette_size != (1 << (c->bpp - 1)))

            return AVERROR_INVALIDDATA;

        avctx->pix_fmt = PIX_FMT_BGR24;

    } else {

        av_log_ask_for_sample(avctx, "unsupported encoding %d and bpp %d\n",

                              encoding, c->bpp);

        return AVERROR_PATCHWELCOME;

    }



    if (p->data[0])

        avctx->release_buffer(avctx, p);



    p->reference = 0;

    if ((ret = avctx->get_buffer(avctx, p)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    p->pict_type = AV_PICTURE_TYPE_I;



    if (encoding) {

        av_fast_padded_malloc(&c->new_video, &c->new_video_size,

                              h * w + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!c->new_video)

            return AVERROR(ENOMEM);

        if (c->bpp == 8)

            cdxl_decode_ham8(c);

        else

            cdxl_decode_ham6(c);

    } else {

        cdxl_decode_rgb(c);

    }

    *data_size      = sizeof(AVFrame);

    *(AVFrame*)data = c->frame;



    return buf_size;

}
