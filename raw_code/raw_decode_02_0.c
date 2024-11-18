static int raw_decode(AVCodecContext *avctx, void *data, int *got_frame,

                      AVPacket *avpkt)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);

    RawVideoContext *context       = avctx->priv_data;

    const uint8_t *buf             = avpkt->data;

    int buf_size                   = avpkt->size;

    int need_copy                  = !avpkt->buf || context->is_2_4_bpp || context->is_yuv2;

    int res;



    AVFrame   *frame   = data;

    AVPicture *picture = data;



    frame->pict_type        = AV_PICTURE_TYPE_I;

    frame->key_frame        = 1;

    frame->reordered_opaque = avctx->reordered_opaque;

    frame->pkt_pts          = avctx->internal->pkt->pts;



    if (buf_size < context->frame_size - (avctx->pix_fmt == AV_PIX_FMT_PAL8 ?

                                          AVPALETTE_SIZE : 0))

        return -1;



    if (need_copy)

        frame->buf[0] = av_buffer_alloc(context->frame_size);

    else

        frame->buf[0] = av_buffer_ref(avpkt->buf);

    if (!frame->buf[0])

        return AVERROR(ENOMEM);



    //2bpp and 4bpp raw in avi and mov (yes this is ugly ...)

    if (context->is_2_4_bpp) {

        int i;

        uint8_t *dst = frame->buf[0]->data;

        buf_size = context->frame_size - AVPALETTE_SIZE;

        if (avctx->bits_per_coded_sample == 4) {

            for (i = 0; 2 * i + 1 < buf_size; i++) {

                dst[2 * i + 0] = buf[i] >> 4;

                dst[2 * i + 1] = buf[i] & 15;

            }

        } else {

            for (i = 0; 4 * i + 3 < buf_size; i++) {

                dst[4 * i + 0] = buf[i] >> 6;

                dst[4 * i + 1] = buf[i] >> 4 & 3;

                dst[4 * i + 2] = buf[i] >> 2 & 3;

                dst[4 * i + 3] = buf[i]      & 3;

            }

        }

        buf = dst;

    } else if (need_copy) {

        memcpy(frame->buf[0]->data, buf, FFMIN(buf_size, context->frame_size));

        buf = frame->buf[0]->data;

    }



    if (avctx->codec_tag == MKTAG('A', 'V', '1', 'x') ||

        avctx->codec_tag == MKTAG('A', 'V', 'u', 'p'))

        buf += buf_size - context->frame_size;



    if ((res = avpicture_fill(picture, buf, avctx->pix_fmt,

                              avctx->width, avctx->height)) < 0)

        return res;



    if (avctx->pix_fmt == AV_PIX_FMT_PAL8) {

        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE,

                                                     NULL);



        if (pal) {

            av_buffer_unref(&context->palette);

            context->palette = av_buffer_alloc(AVPALETTE_SIZE);

            if (!context->palette)

                return AVERROR(ENOMEM);

            memcpy(context->palette->data, pal, AVPALETTE_SIZE);

            frame->palette_has_changed = 1;

        }

    }



    if ((avctx->pix_fmt == AV_PIX_FMT_PAL8 && buf_size < context->frame_size) ||

        (desc->flags & AV_PIX_FMT_FLAG_PSEUDOPAL)) {

        frame->buf[1]  = av_buffer_ref(context->palette);

        if (!frame->buf[1])

            return AVERROR(ENOMEM);

        frame->data[1] = frame->buf[1]->data;

    }

    if (avctx->pix_fmt == AV_PIX_FMT_BGR24 &&

        ((frame->linesize[0] + 3) & ~3) * avctx->height <= buf_size)

        frame->linesize[0] = (frame->linesize[0] + 3) & ~3;



    if (context->flip)

        flip(avctx, picture);



    if (avctx->codec_tag == MKTAG('Y', 'V', '1', '2') ||

        avctx->codec_tag == MKTAG('Y', 'V', '1', '6') ||

        avctx->codec_tag == MKTAG('Y', 'V', '2', '4') ||

        avctx->codec_tag == MKTAG('Y', 'V', 'U', '9'))

        FFSWAP(uint8_t *, picture->data[1], picture->data[2]);



    if (avctx->codec_tag == AV_RL32("yuv2") &&

        avctx->pix_fmt   == AV_PIX_FMT_YUYV422) {

        int x, y;

        uint8_t *line = picture->data[0];

        for (y = 0; y < avctx->height; y++) {

            for (x = 0; x < avctx->width; x++)

                line[2 * x + 1] ^= 0x80;

            line += picture->linesize[0];

        }

    }



    *got_frame = 1;

    return buf_size;

}
