static int libkvazaar_encode(AVCodecContext *avctx,

                             AVPacket *avpkt,

                             const AVFrame *frame,

                             int *got_packet_ptr)

{

    int retval = 0;

    kvz_picture *img_in = NULL;



    kvz_data_chunk *data_out = NULL;

    uint32_t len_out = 0;

    kvz_frame_info frame_info;



    LibkvazaarContext *ctx = avctx->priv_data;



    *got_packet_ptr = 0;



    if (frame) {

        int i = 0;



        av_assert0(frame->width == ctx->config->width);

        av_assert0(frame->height == ctx->config->height);

        av_assert0(frame->format == avctx->pix_fmt);



        // Allocate input picture for kvazaar.

        img_in = ctx->api->picture_alloc(frame->width, frame->height);

        if (!img_in) {

            av_log(avctx, AV_LOG_ERROR, "Failed to allocate picture.\n");

            retval = AVERROR(ENOMEM);

            goto done;

        }



        // Copy pixels from frame to img_in.

        for (i = 0; i < 3; ++i) {

            uint8_t *dst = img_in->data[i];

            uint8_t *src = frame->data[i];

            int width = (i == 0) ? frame->width : (frame->width / 2);

            int height = (i == 0) ? frame->height : (frame->height / 2);

            int y = 0;

            for (y = 0; y < height; ++y) {

                memcpy(dst, src, width);

                src += frame->linesize[i];

                dst += width;

            }

        }

    }



    if (!ctx->api->encoder_encode(ctx->encoder, img_in,

                                  &data_out, &len_out,

                                  NULL, NULL,

                                  &frame_info)) {

        av_log(avctx, AV_LOG_ERROR, "Failed to encode frame.\n");

        retval = AVERROR_EXTERNAL;

        goto done;

    }



    if (data_out) {

        kvz_data_chunk *chunk = NULL;

        uint64_t written = 0;



        retval = ff_alloc_packet(avpkt, len_out);

        if (retval < 0) {

            av_log(avctx, AV_LOG_ERROR, "Failed to allocate output packet.\n");

            goto done;

        }



        for (chunk = data_out; chunk != NULL; chunk = chunk->next) {

            av_assert0(written + chunk->len <= len_out);

            memcpy(avpkt->data + written, chunk->data, chunk->len);

            written += chunk->len;

        }

        *got_packet_ptr = 1;



        ctx->api->chunk_free(data_out);

        data_out = NULL;



        avpkt->flags = 0;

        // IRAP VCL NAL unit types span the range

        // [BLA_W_LP (16), RSV_IRAP_VCL23 (23)].

        if (frame_info.nal_unit_type >= KVZ_NAL_BLA_W_LP &&

                frame_info.nal_unit_type <= KVZ_NAL_RSV_IRAP_VCL23) {

            avpkt->flags |= AV_PKT_FLAG_KEY;

        }

    }



done:

    ctx->api->picture_free(img_in);

    ctx->api->chunk_free(data_out);

    return retval;

}
