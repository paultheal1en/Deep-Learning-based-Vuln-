static int decode_frame(AVCodecContext *avctx, void *data, int *data_size,

                        AVPacket *avpkt)

{

    V210DecContext *s = avctx->priv_data;



    int h, w, stride, aligned_input;

    AVFrame *pic = avctx->coded_frame;

    const uint8_t *psrc = avpkt->data;

    uint16_t *y, *u, *v;



    if (s->custom_stride )

        stride = s->custom_stride;

    else {

        int aligned_width = ((avctx->width + 47) / 48) * 48;

        stride = aligned_width * 8 / 3;

    }



    aligned_input = !((uintptr_t)psrc & 0xf) && !(stride & 0xf);

    if (aligned_input != s->aligned_input) {

        s->aligned_input = aligned_input;

        if (HAVE_MMX)

            v210_x86_init(s);

    }



    if (pic->data[0])

        avctx->release_buffer(avctx, pic);



    if (avpkt->size < stride * avctx->height) {

        av_log(avctx, AV_LOG_ERROR, "packet too small\n");

        return -1;

    }



    pic->reference = 0;

    if (avctx->get_buffer(avctx, pic) < 0)

        return -1;



    y = (uint16_t*)pic->data[0];

    u = (uint16_t*)pic->data[1];

    v = (uint16_t*)pic->data[2];

    pic->pict_type = AV_PICTURE_TYPE_I;

    pic->key_frame = 1;



    for (h = 0; h < avctx->height; h++) {

        const uint32_t *src = (const uint32_t*)psrc;

        uint32_t val;



        w = (avctx->width / 6) * 6;

        s->unpack_frame(src, y, u, v, w);



        y += w;

        u += w >> 1;

        v += w >> 1;

        src += (w << 1) / 3;



        if (w < avctx->width - 1) {

            READ_PIXELS(u, y, v);



            val  = av_le2ne32(*src++);

            *y++ =  val & 0x3FF;

        }

        if (w < avctx->width - 3) {

            *u++ = (val >> 10) & 0x3FF;

            *y++ = (val >> 20) & 0x3FF;



            val  = av_le2ne32(*src++);

            *v++ =  val & 0x3FF;

            *y++ = (val >> 10) & 0x3FF;

        }



        psrc += stride;

        y += pic->linesize[0] / 2 - avctx->width;

        u += pic->linesize[1] / 2 - avctx->width / 2;

        v += pic->linesize[2] / 2 - avctx->width / 2;

    }



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = *avctx->coded_frame;



    return avpkt->size;

}
