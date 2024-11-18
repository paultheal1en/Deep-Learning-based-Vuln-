static int decode_packet(int *got_frame, int cached)

{

    int ret = 0;

    int decoded = pkt.size;



    *got_frame = 0;



    if (pkt.stream_index == video_stream_idx) {

        /* decode video frame */

        ret = avcodec_decode_video2(video_dec_ctx, frame, got_frame, &pkt);

        if (ret < 0) {

            fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));

            return ret;

        }

        if (video_dec_ctx->width != width || video_dec_ctx->height != height ||

            video_dec_ctx->pix_fmt != pix_fmt) {

            /* To handle this change, one could call av_image_alloc again and

             * decode the following frames into another rawvideo file. */

            fprintf(stderr, "Error: Width, height and pixel format have to be "

                    "constant in a rawvideo file, but the width, height or "

                    "pixel format of the input video changed:\n"

                    "old: width = %d, height = %d, format = %s\n"

                    "new: width = %d, height = %d, format = %s\n",

                    width, height, av_get_pix_fmt_name(pix_fmt),

                    video_dec_ctx->width, video_dec_ctx->height,

                    av_get_pix_fmt_name(video_dec_ctx->pix_fmt));

            return -1;

        }



        if (*got_frame) {

            printf("video_frame%s n:%d coded_n:%d pts:%s\n",

                   cached ? "(cached)" : "",

                   video_frame_count++, frame->coded_picture_number,

                   av_ts2timestr(frame->pts, &video_dec_ctx->time_base));



            /* copy decoded frame to destination buffer:

             * this is required since rawvideo expects non aligned data */

            av_image_copy(video_dst_data, video_dst_linesize,

                          (const uint8_t **)(frame->data), frame->linesize,

                          pix_fmt, width, height);



            /* write to rawvideo file */

            fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);

        }

    } else if (pkt.stream_index == audio_stream_idx) {

        /* decode audio frame */

        ret = avcodec_decode_audio4(audio_dec_ctx, frame, got_frame, &pkt);

        if (ret < 0) {

            fprintf(stderr, "Error decoding audio frame (%s)\n", av_err2str(ret));

            return ret;

        }

        /* Some audio decoders decode only part of the packet, and have to be

         * called again with the remainder of the packet data.

         * Sample: fate-suite/lossless-audio/luckynight-partial.shn

         * Also, some decoders might over-read the packet. */

        decoded = FFMIN(ret, pkt.size);



        if (*got_frame) {

            size_t unpadded_linesize = frame->nb_samples * av_get_bytes_per_sample(frame->format);

            printf("audio_frame%s n:%d nb_samples:%d pts:%s\n",

                   cached ? "(cached)" : "",

                   audio_frame_count++, frame->nb_samples,

                   av_ts2timestr(frame->pts, &audio_dec_ctx->time_base));



            /* Write the raw audio data samples of the first plane. This works

             * fine for packed formats (e.g. AV_SAMPLE_FMT_S16). However,

             * most audio decoders output planar audio, which uses a separate

             * plane of audio samples for each channel (e.g. AV_SAMPLE_FMT_S16P).

             * In other words, this code will write only the first audio channel

             * in these cases.

             * You should use libswresample or libavfilter to convert the frame

             * to packed data. */

            fwrite(frame->extended_data[0], 1, unpadded_linesize, audio_dst_file);

        }

    }



    /* If we use the new API with reference counting, we own the data and need

     * to de-reference it when we don't use it anymore */

    if (*got_frame && api_mode == API_MODE_NEW_API_REF_COUNT)

        av_frame_unref(frame);



    return decoded;

}
