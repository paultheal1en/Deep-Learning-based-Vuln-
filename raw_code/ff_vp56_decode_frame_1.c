int ff_vp56_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                         AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    VP56Context *s = avctx->priv_data;

    AVFrame *const p = s->frames[VP56_FRAME_CURRENT];

    int remaining_buf_size = avpkt->size;

    int av_uninit(alpha_offset);

    int i, res;

    int ret;



    if (s->has_alpha) {

        if (remaining_buf_size < 3)

            return AVERROR_INVALIDDATA;

        alpha_offset = bytestream_get_be24(&buf);

        remaining_buf_size -= 3;

        if (remaining_buf_size < alpha_offset)

            return AVERROR_INVALIDDATA;

    }



    res = s->parse_header(s, buf, remaining_buf_size);

    if (res < 0)

        return res;



    if (res == VP56_SIZE_CHANGE) {

        for (i = 0; i < 4; i++) {

            av_frame_unref(s->frames[i]);

            if (s->alpha_context)

                av_frame_unref(s->alpha_context->frames[i]);

        }

    }



    ret = ff_get_buffer(avctx, p, AV_GET_BUFFER_FLAG_REF);

    if (ret < 0)

        return ret;



    if (avctx->pix_fmt == AV_PIX_FMT_YUVA420P) {

        av_frame_unref(s->alpha_context->frames[VP56_FRAME_CURRENT]);

        if ((ret = av_frame_ref(s->alpha_context->frames[VP56_FRAME_CURRENT], p)) < 0) {

            av_frame_unref(p);

            return ret;

        }

    }



    if (res == VP56_SIZE_CHANGE) {

        if (vp56_size_changed(s)) {

            av_frame_unref(p);

            return AVERROR_INVALIDDATA;

        }

    }



    if (avctx->pix_fmt == AV_PIX_FMT_YUVA420P) {

        int bak_w = avctx->width;

        int bak_h = avctx->height;

        int bak_cw = avctx->coded_width;

        int bak_ch = avctx->coded_height;

        buf += alpha_offset;

        remaining_buf_size -= alpha_offset;



        res = s->alpha_context->parse_header(s->alpha_context, buf, remaining_buf_size);

        if (res != 0) {

            if(res==VP56_SIZE_CHANGE) {

                av_log(avctx, AV_LOG_ERROR, "Alpha reconfiguration\n");

                avctx->width  = bak_w;

                avctx->height = bak_h;

                avctx->coded_width  = bak_cw;

                avctx->coded_height = bak_ch;

            }

            av_frame_unref(p);

            return AVERROR_INVALIDDATA;

        }

    }



    avctx->execute2(avctx, ff_vp56_decode_mbs, 0, 0, (avctx->pix_fmt == AV_PIX_FMT_YUVA420P) + 1);



    if ((res = av_frame_ref(data, p)) < 0)

        return res;

    *got_frame = 1;



    return avpkt->size;

}
