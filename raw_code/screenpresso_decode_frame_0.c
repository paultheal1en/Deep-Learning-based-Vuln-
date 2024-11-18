static int screenpresso_decode_frame(AVCodecContext *avctx, void *data,

                                     int *got_frame, AVPacket *avpkt)

{

    ScreenpressoContext *ctx = avctx->priv_data;

    AVFrame *frame = data;

    int keyframe;

    int ret;



    /* Size check */

    if (avpkt->size < 3) {

        av_log(avctx, AV_LOG_ERROR, "Packet too small (%d)\n", avpkt->size);

        return AVERROR_INVALIDDATA;

    }



    /* Basic sanity check, but not really harmful */

    if ((avpkt->data[0] != 0x73 && avpkt->data[0] != 0x72) ||

        avpkt->data[1] != 8) { // bpp probably

        av_log(avctx, AV_LOG_WARNING, "Unknown header 0x%02X%02X\n",

               avpkt->data[0], avpkt->data[1]);

    }

    keyframe = (avpkt->data[0] == 0x73);



    /* Resize deflate buffer and frame on resolution change */

    if (ctx->inflated_size != avctx->width * avctx->height * 3) {

        av_frame_unref(ctx->current);

        ret = ff_get_buffer(avctx, ctx->current, AV_GET_BUFFER_FLAG_REF);

        if (ret < 0)

            return ret;



        /* If malloc fails, reset len to avoid preserving an invalid value */

        ctx->inflated_size = avctx->width * avctx->height * 3;

        ret = av_reallocp(&ctx->inflated_buf, ctx->inflated_size);

        if (ret < 0) {

            ctx->inflated_size = 0;

            return ret;

        }

    }



    /* Inflate the frame after the 2 byte header */

    ret = uncompress(ctx->inflated_buf, &ctx->inflated_size,

                     avpkt->data + 2, avpkt->size - 2);

    if (ret) {

        av_log(avctx, AV_LOG_ERROR, "Deflate error %d.\n", ret);

        return AVERROR_UNKNOWN;

    }



    /* When a keyframe is found, copy it (flipped) */

    if (keyframe)

        av_image_copy_plane(ctx->current->data[0] +

                            ctx->current->linesize[0] * (avctx->height - 1),

                            -1 * ctx->current->linesize[0],

                            ctx->inflated_buf, avctx->width * 3,

                            avctx->width * 3, avctx->height);

    /* Otherwise sum the delta on top of the current frame */

    else

        sum_delta_flipped(ctx->current->data[0], ctx->current->linesize[0],

                          ctx->inflated_buf, avctx->width * 3,

                          avctx->width * 3, avctx->height);



    /* Frame is ready to be output */

    ret = av_frame_ref(frame, ctx->current);

    if (ret < 0)

        return ret;



    /* Usual properties */

    if (keyframe) {

        frame->pict_type = AV_PICTURE_TYPE_I;

        frame->key_frame = 1;

    } else {

        frame->pict_type = AV_PICTURE_TYPE_P;

    }

    *got_frame = 1;



    return 0;

}
