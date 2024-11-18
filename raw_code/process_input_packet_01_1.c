static void process_input_packet(InputStream *ist, const AVPacket *pkt, int no_eof)

{

    int i;

    int repeating = 0;

    AVPacket avpkt;



    if (ist->next_dts == AV_NOPTS_VALUE)

        ist->next_dts = ist->last_dts;



    if (!pkt) {

        /* EOF handling */

        av_init_packet(&avpkt);

        avpkt.data = NULL;

        avpkt.size = 0;

    } else {

        avpkt = *pkt;

    }



    if (pkt && pkt->dts != AV_NOPTS_VALUE)

        ist->next_dts = ist->last_dts = av_rescale_q(pkt->dts, ist->st->time_base, AV_TIME_BASE_Q);



    // while we have more to decode or while the decoder did output something on EOF

    while (ist->decoding_needed && (!pkt || avpkt.size > 0)) {

        int ret = 0;

        int got_output = 0;



        if (!repeating)

            ist->last_dts = ist->next_dts;



        switch (ist->dec_ctx->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ret = decode_audio    (ist, repeating ? NULL : &avpkt, &got_output);

            break;

        case AVMEDIA_TYPE_VIDEO:

            ret = decode_video    (ist, repeating ? NULL : &avpkt, &got_output);

            if (repeating && !got_output)

                ;

            else if (pkt && pkt->duration)

                ist->next_dts += av_rescale_q(pkt->duration, ist->st->time_base, AV_TIME_BASE_Q);

            else if (ist->st->avg_frame_rate.num)

                ist->next_dts += av_rescale_q(1, av_inv_q(ist->st->avg_frame_rate),

                                              AV_TIME_BASE_Q);

            else if (ist->dec_ctx->framerate.num != 0) {

                int ticks      = ist->st->parser ? ist->st->parser->repeat_pict + 1 :

                                                   ist->dec_ctx->ticks_per_frame;

                ist->next_dts += av_rescale_q(ticks, ist->dec_ctx->framerate, AV_TIME_BASE_Q);

            }

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            if (repeating)

                break;

            ret = transcode_subtitles(ist, &avpkt, &got_output);

            break;

        default:

            return;

        }



        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error while decoding stream #%d:%d\n",

                   ist->file_index, ist->st->index);

            if (exit_on_error)

                exit_program(1);

            break;

        }



        if (!got_output)

            break;



        repeating = 1;

    }



    /* after flushing, send an EOF on all the filter inputs attached to the stream */

    /* except when looping we need to flush but not to send an EOF */

    if (!pkt && ist->decoding_needed && !no_eof) {

        int ret = send_filter_eof(ist);

        if (ret < 0) {

            av_log(NULL, AV_LOG_FATAL, "Error marking filters as finished\n");

            exit_program(1);

        }

    }



    /* handle stream copy */

    if (!ist->decoding_needed) {

        ist->last_dts = ist->next_dts;

        switch (ist->dec_ctx->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ist->next_dts += ((int64_t)AV_TIME_BASE * ist->dec_ctx->frame_size) /

                             ist->dec_ctx->sample_rate;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (ist->dec_ctx->framerate.num != 0) {

                int ticks = ist->st->parser ? ist->st->parser->repeat_pict + 1 : ist->dec_ctx->ticks_per_frame;

                ist->next_dts += ((int64_t)AV_TIME_BASE *

                                  ist->dec_ctx->framerate.den * ticks) /

                                  ist->dec_ctx->framerate.num;

            }

            break;

        }

    }

    for (i = 0; pkt && i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];



        if (!check_output_constraints(ist, ost) || ost->encoding_needed)

            continue;



        do_streamcopy(ist, ost, pkt);

    }



    return;

}
