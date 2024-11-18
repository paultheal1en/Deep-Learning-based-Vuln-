static int libschroedinger_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                                        const AVFrame *frame, int *got_packet)

{

    int enc_size = 0;

    SchroEncoderParams *p_schro_params = avctx->priv_data;

    SchroEncoder *encoder = p_schro_params->encoder;

    struct FFSchroEncodedFrame *p_frame_output = NULL;

    int go = 1;

    SchroBuffer *enc_buf;

    int presentation_frame;

    int parse_code;

    int last_frame_in_sequence = 0;

    int pkt_size, ret;



    if (!frame) {

        /* Push end of sequence if not already signalled. */

        if (!p_schro_params->eos_signalled) {

            schro_encoder_end_of_stream(encoder);

            p_schro_params->eos_signalled = 1;

        }

    } else {

        /* Allocate frame data to schro input buffer. */

        SchroFrame *in_frame = libschroedinger_frame_from_data(avctx, frame);

        if (!in_frame)

            return AVERROR(ENOMEM);

        /* Load next frame. */

        schro_encoder_push_frame(encoder, in_frame);

    }



    if (p_schro_params->eos_pulled)

        go = 0;



    /* Now check to see if we have any output from the encoder. */

    while (go) {

        int err;

        SchroStateEnum state;

        state = schro_encoder_wait(encoder);

        switch (state) {

        case SCHRO_STATE_HAVE_BUFFER:

        case SCHRO_STATE_END_OF_STREAM:

            enc_buf = schro_encoder_pull(encoder, &presentation_frame);

            if (enc_buf->length <= 0)

                return AVERROR_BUG;

            parse_code = enc_buf->data[4];



            /* All non-frame data is prepended to actual frame data to

             * be able to set the pts correctly. So we don't write data

             * to the frame output queue until we actually have a frame

             */

            if ((err = av_reallocp(&p_schro_params->enc_buf,

                                   p_schro_params->enc_buf_size +

                                   enc_buf->length)) < 0) {

                p_schro_params->enc_buf_size = 0;

                return err;

            }



            memcpy(p_schro_params->enc_buf + p_schro_params->enc_buf_size,

                   enc_buf->data, enc_buf->length);

            p_schro_params->enc_buf_size += enc_buf->length;





            if (state == SCHRO_STATE_END_OF_STREAM) {

                p_schro_params->eos_pulled = 1;

                go = 0;

            }



            if (!SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {

                schro_buffer_unref(enc_buf);

                break;

            }



            /* Create output frame. */

            p_frame_output = av_mallocz(sizeof(FFSchroEncodedFrame));

            if (!p_frame_output)

                return AVERROR(ENOMEM);

            /* Set output data. */

            p_frame_output->size     = p_schro_params->enc_buf_size;

            p_frame_output->p_encbuf = p_schro_params->enc_buf;

            if (SCHRO_PARSE_CODE_IS_INTRA(parse_code) &&

                SCHRO_PARSE_CODE_IS_REFERENCE(parse_code))

                p_frame_output->key_frame = 1;



            /* Parse the coded frame number from the bitstream. Bytes 14

             * through 17 represent the frame number. */

            p_frame_output->frame_num = AV_RB32(enc_buf->data + 13);



            ff_schro_queue_push_back(&p_schro_params->enc_frame_queue,

                                     p_frame_output);

            p_schro_params->enc_buf_size = 0;

            p_schro_params->enc_buf      = NULL;



            schro_buffer_unref(enc_buf);



            break;



        case SCHRO_STATE_NEED_FRAME:

            go = 0;

            break;



        case SCHRO_STATE_AGAIN:

            break;



        default:

            av_log(avctx, AV_LOG_ERROR, "Unknown Schro Encoder state\n");

            return -1;

        }

    }



    /* Copy 'next' frame in queue. */



    if (p_schro_params->enc_frame_queue.size == 1 &&

        p_schro_params->eos_pulled)

        last_frame_in_sequence = 1;



    p_frame_output = ff_schro_queue_pop(&p_schro_params->enc_frame_queue);



    if (!p_frame_output)

        return 0;



    pkt_size = p_frame_output->size;

    if (last_frame_in_sequence && p_schro_params->enc_buf_size > 0)

        pkt_size += p_schro_params->enc_buf_size;

    if ((ret = ff_alloc_packet2(avctx, pkt, pkt_size, 0)) < 0)

        goto error;



    memcpy(pkt->data, p_frame_output->p_encbuf, p_frame_output->size);

#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    avctx->coded_frame->key_frame = p_frame_output->key_frame;

    avctx->coded_frame->pts = p_frame_output->frame_num;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    /* Use the frame number of the encoded frame as the pts. It is OK to

     * do so since Dirac is a constant frame rate codec. It expects input

     * to be of constant frame rate. */

    pkt->pts = p_frame_output->frame_num;

    pkt->dts = p_schro_params->dts++;

    enc_size = p_frame_output->size;



    /* Append the end of sequence information to the last frame in the

     * sequence. */

    if (last_frame_in_sequence && p_schro_params->enc_buf_size > 0) {

        memcpy(pkt->data + enc_size, p_schro_params->enc_buf,

               p_schro_params->enc_buf_size);

        enc_size += p_schro_params->enc_buf_size;

        av_freep(&p_schro_params->enc_buf);

        p_schro_params->enc_buf_size = 0;

    }



    if (p_frame_output->key_frame)

        pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



error:

    /* free frame */

    libschroedinger_free_frame(p_frame_output);

    return ret;

}
