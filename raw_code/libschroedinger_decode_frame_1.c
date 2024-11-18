static int libschroedinger_decode_frame(AVCodecContext *avctx,

                                        void *data, int *got_frame,

                                        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    int64_t pts  = avpkt->pts;

    SchroTag *tag;



    SchroDecoderParams *p_schro_params = avctx->priv_data;

    SchroDecoder *decoder = p_schro_params->decoder;

    SchroBuffer *enc_buf;

    SchroFrame* frame;

    AVFrame *avframe = data;

    int state;

    int go = 1;

    int outer = 1;

    SchroParseUnitContext parse_ctx;

    LibSchroFrameContext *framewithpts = NULL;

    int ret;



    *got_frame = 0;



    parse_context_init(&parse_ctx, buf, buf_size);

    if (!buf_size) {

        if (!p_schro_params->eos_signalled) {

            state = schro_decoder_push_end_of_stream(decoder);

            p_schro_params->eos_signalled = 1;

        }

    }



    /* Loop through all the individual parse units in the input buffer */

    do {

        if ((enc_buf = find_next_parse_unit(&parse_ctx))) {

            /* Set Schrotag with the pts to be recovered after decoding*/

            enc_buf->tag = schro_tag_new(av_malloc(sizeof(int64_t)), av_free);

            if (!enc_buf->tag->value) {

                av_log(avctx, AV_LOG_ERROR, "Unable to allocate SchroTag\n");

                return AVERROR(ENOMEM);

            }

            AV_WN(64, enc_buf->tag->value, pts);

            /* Push buffer into decoder. */

            if (SCHRO_PARSE_CODE_IS_PICTURE(enc_buf->data[4]) &&

                SCHRO_PARSE_CODE_NUM_REFS(enc_buf->data[4]) > 0)

                avctx->has_b_frames = 1;

            state = schro_decoder_push(decoder, enc_buf);

            if (state == SCHRO_DECODER_FIRST_ACCESS_UNIT)

                libschroedinger_handle_first_access_unit(avctx);

            go = 1;

        } else

            outer = 0;



        while (go) {

            /* Parse data and process result. */

            state = schro_decoder_wait(decoder);

            switch (state) {

            case SCHRO_DECODER_FIRST_ACCESS_UNIT:

                libschroedinger_handle_first_access_unit(avctx);

                break;



            case SCHRO_DECODER_NEED_BITS:

                /* Need more input data - stop iterating over what we have. */

                go = 0;

                break;



            case SCHRO_DECODER_NEED_FRAME:

                /* Decoder needs a frame - create one and push it in. */

                frame = ff_create_schro_frame(avctx,

                                              p_schro_params->frame_format);

                if (!frame)

                    return AVERROR(ENOMEM);

                schro_decoder_add_output_picture(decoder, frame);

                break;



            case SCHRO_DECODER_OK:

                /* Pull a frame out of the decoder. */

                tag   = schro_decoder_get_picture_tag(decoder);

                frame = schro_decoder_pull(decoder);



                if (frame) {

                    /* Add relation between schroframe and pts. */

                    framewithpts = av_malloc(sizeof(LibSchroFrameContext));

                    if (!framewithpts) {

                        av_log(avctx, AV_LOG_ERROR, "Unable to allocate FrameWithPts\n");

                        return AVERROR(ENOMEM);

                    }

                    framewithpts->frame = frame;

                    framewithpts->pts   = AV_RN64(tag->value);

                    ff_schro_queue_push_back(&p_schro_params->dec_frame_queue,

                                             framewithpts);

                }

                break;

            case SCHRO_DECODER_EOS:

                go = 0;

                p_schro_params->eos_pulled = 1;

                schro_decoder_reset(decoder);

                outer = 0;

                break;



            case SCHRO_DECODER_ERROR:

                return -1;

                break;

            }

        }

    } while (outer);



    /* Grab next frame to be returned from the top of the queue. */

    framewithpts = ff_schro_queue_pop(&p_schro_params->dec_frame_queue);



    if (framewithpts && framewithpts->frame && framewithpts->frame->components[0].stride) {

        if ((ret = ff_get_buffer(avctx, avframe, 0)) < 0) {

            goto end;

        }



        memcpy(avframe->data[0],

               framewithpts->frame->components[0].data,

               framewithpts->frame->components[0].length);



        memcpy(avframe->data[1],

               framewithpts->frame->components[1].data,

               framewithpts->frame->components[1].length);



        memcpy(avframe->data[2],

               framewithpts->frame->components[2].data,

               framewithpts->frame->components[2].length);



        /* Fill frame with current buffer data from Schroedinger. */

        avframe->pts = framewithpts->pts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

        avframe->pkt_pts = avframe->pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

        avframe->linesize[0] = framewithpts->frame->components[0].stride;

        avframe->linesize[1] = framewithpts->frame->components[1].stride;

        avframe->linesize[2] = framewithpts->frame->components[2].stride;



        *got_frame      = 1;

    } else {

        data       = NULL;

        *got_frame = 0;

    }

    ret = buf_size;

end:

    /* Now free the frame resources. */

    if (framewithpts && framewithpts->frame)

        libschroedinger_decode_frame_free(framewithpts->frame);

    av_freep(&framewithpts);

    return ret;

}
