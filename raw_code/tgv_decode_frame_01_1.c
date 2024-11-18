static int tgv_decode_frame(AVCodecContext *avctx,

                            void *data, int *got_frame,

                            AVPacket *avpkt)

{

    const uint8_t *buf     = avpkt->data;

    int buf_size           = avpkt->size;

    TgvContext *s          = avctx->priv_data;

    const uint8_t *buf_end = buf + buf_size;

    AVFrame *frame         = data;

    int chunk_type, ret;



    if (buf_end - buf < EA_PREAMBLE_SIZE)

        return AVERROR_INVALIDDATA;



    chunk_type = AV_RL32(&buf[0]);

    buf       += EA_PREAMBLE_SIZE;



    if (chunk_type == kVGT_TAG) {

        int pal_count, i;

        if(buf_end - buf < 12) {

            av_log(avctx, AV_LOG_WARNING, "truncated header\n");

            return AVERROR_INVALIDDATA;

        }



        s->width  = AV_RL16(&buf[0]);

        s->height = AV_RL16(&buf[2]);

        if (s->avctx->width != s->width || s->avctx->height != s->height) {

            av_freep(&s->frame_buffer);

            av_frame_unref(s->last_frame);

            if ((ret = ff_set_dimensions(s->avctx, s->width, s->height)) < 0)

                return ret;

        }



        pal_count = AV_RL16(&buf[6]);

        buf += 12;

        for(i = 0; i < pal_count && i < AVPALETTE_COUNT && buf_end - buf >= 3; i++) {

            s->palette[i] = 0xFFU << 24 | AV_RB24(buf);

            buf += 3;

        }

    }



    if ((ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)

        return ret;



    memcpy(frame->data[1], s->palette, AVPALETTE_SIZE);



    if (chunk_type == kVGT_TAG) {

        int y;

        frame->key_frame = 1;

        frame->pict_type = AV_PICTURE_TYPE_I;



        if (!s->frame_buffer &&

            !(s->frame_buffer = av_malloc(s->width * s->height)))

            return AVERROR(ENOMEM);



        if (unpack(buf, buf_end, s->frame_buffer, s->avctx->width, s->avctx->height) < 0) {

            av_log(avctx, AV_LOG_WARNING, "truncated intra frame\n");

            return AVERROR_INVALIDDATA;

        }

        for (y = 0; y < s->height; y++)

            memcpy(frame->data[0]  + y * frame->linesize[0],

                   s->frame_buffer + y * s->width,

                   s->width);

    } else {

        if (!s->last_frame->data[0]) {

            av_log(avctx, AV_LOG_WARNING, "inter frame without corresponding intra frame\n");

            return buf_size;

        }

        frame->key_frame = 0;

        frame->pict_type = AV_PICTURE_TYPE_P;

        if (tgv_decode_inter(s, frame, buf, buf_end) < 0) {

            av_log(avctx, AV_LOG_WARNING, "truncated inter frame\n");

            return AVERROR_INVALIDDATA;

        }

    }



    av_frame_unref(s->last_frame);

    if ((ret = av_frame_ref(s->last_frame, frame)) < 0)

        return ret;



    *got_frame = 1;



    return buf_size;

}
