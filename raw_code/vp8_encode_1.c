static int vp8_encode(AVCodecContext *avctx, AVPacket *pkt,

                      const AVFrame *frame, int *got_packet)

{

    VP8Context *ctx = avctx->priv_data;

    struct vpx_image *rawimg = NULL;

    struct vpx_image *rawimg_alpha = NULL;

    int64_t timestamp = 0;

    int res, coded_size;

    vpx_enc_frame_flags_t flags = 0;



    if (frame) {

        rawimg                      = &ctx->rawimg;

        rawimg->planes[VPX_PLANE_Y] = frame->data[0];

        rawimg->planes[VPX_PLANE_U] = frame->data[1];

        rawimg->planes[VPX_PLANE_V] = frame->data[2];

        rawimg->stride[VPX_PLANE_Y] = frame->linesize[0];

        rawimg->stride[VPX_PLANE_U] = frame->linesize[1];

        rawimg->stride[VPX_PLANE_V] = frame->linesize[2];

        if (ctx->is_alpha) {

            uint8_t *u_plane, *v_plane;

            rawimg_alpha = &ctx->rawimg_alpha;

            rawimg_alpha->planes[VPX_PLANE_Y] = frame->data[3];

            u_plane = av_malloc(frame->linesize[1] * frame->height);

            memset(u_plane, 0x80, frame->linesize[1] * frame->height);

            rawimg_alpha->planes[VPX_PLANE_U] = u_plane;

            v_plane = av_malloc(frame->linesize[2] * frame->height);

            memset(v_plane, 0x80, frame->linesize[2] * frame->height);

            rawimg_alpha->planes[VPX_PLANE_V] = v_plane;

            rawimg_alpha->stride[VPX_PLANE_Y] = frame->linesize[0];

            rawimg_alpha->stride[VPX_PLANE_U] = frame->linesize[1];

            rawimg_alpha->stride[VPX_PLANE_V] = frame->linesize[2];

        }

        timestamp                   = frame->pts;

        if (frame->pict_type == AV_PICTURE_TYPE_I)

            flags |= VPX_EFLAG_FORCE_KF;

    }



    res = vpx_codec_encode(&ctx->encoder, rawimg, timestamp,

                           avctx->ticks_per_frame, flags, ctx->deadline);

    if (res != VPX_CODEC_OK) {

        log_encoder_error(avctx, "Error encoding frame");

        return AVERROR_INVALIDDATA;

    }



    if (ctx->is_alpha) {

        res = vpx_codec_encode(&ctx->encoder_alpha, rawimg_alpha, timestamp,

                               avctx->ticks_per_frame, flags, ctx->deadline);

        if (res != VPX_CODEC_OK) {

            log_encoder_error(avctx, "Error encoding alpha frame");

            return AVERROR_INVALIDDATA;

        }

    }



    coded_size = queue_frames(avctx, pkt, avctx->coded_frame);



    if (!frame && avctx->flags & CODEC_FLAG_PASS1) {

        unsigned int b64_size = AV_BASE64_SIZE(ctx->twopass_stats.sz);



        avctx->stats_out = av_malloc(b64_size);

        if (!avctx->stats_out) {

            av_log(avctx, AV_LOG_ERROR, "Stat buffer alloc (%d bytes) failed\n",

                   b64_size);

            return AVERROR(ENOMEM);

        }

        av_base64_encode(avctx->stats_out, b64_size, ctx->twopass_stats.buf,

                         ctx->twopass_stats.sz);

    }



    if (rawimg_alpha) {

        av_freep(&rawimg_alpha->planes[VPX_PLANE_U]);

        av_freep(&rawimg_alpha->planes[VPX_PLANE_V]);

    }



    *got_packet = !!coded_size;

    return 0;

}
