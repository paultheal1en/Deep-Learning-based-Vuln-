static void flush_encoders(void)

{

    int i, ret;



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream   *ost = output_streams[i];

        AVCodecContext *enc = ost->enc_ctx;

        OutputFile      *of = output_files[ost->file_index];



        if (!ost->encoding_needed)

            continue;



        // Try to enable encoding with no input frames.

        // Maybe we should just let encoding fail instead.

        if (!ost->initialized) {

            FilterGraph *fg = ost->filter->graph;

            char error[1024];



            av_log(NULL, AV_LOG_WARNING,

                   "Finishing stream %d:%d without any data written to it.\n",

                   ost->file_index, ost->st->index);



            if (ost->filter && !fg->graph) {

                int x;

                for (x = 0; x < fg->nb_inputs; x++) {

                    InputFilter *ifilter = fg->inputs[x];

                    if (ifilter->format < 0) {

                        AVCodecParameters *par = ifilter->ist->st->codecpar;

                        // We never got any input. Set a fake format, which will

                        // come from libavformat.

                        ifilter->format                 = par->format;

                        ifilter->sample_rate            = par->sample_rate;

                        ifilter->channels               = par->channels;

                        ifilter->channel_layout         = par->channel_layout;

                        ifilter->width                  = par->width;

                        ifilter->height                 = par->height;

                        ifilter->sample_aspect_ratio    = par->sample_aspect_ratio;

                    }

                }



                if (!ifilter_has_all_input_formats(fg))

                    continue;



                ret = configure_filtergraph(fg);

                if (ret < 0) {

                    av_log(NULL, AV_LOG_ERROR, "Error configuring filter graph\n");

                    exit_program(1);

                }



                finish_output_stream(ost);

            }



            ret = init_output_stream(ost, error, sizeof(error));

            if (ret < 0) {

                av_log(NULL, AV_LOG_ERROR, "Error initializing output stream %d:%d -- %s\n",

                       ost->file_index, ost->index, error);

                exit_program(1);

            }

        }



        if (enc->codec_type == AVMEDIA_TYPE_AUDIO && enc->frame_size <= 1)

            continue;

#if FF_API_LAVF_FMT_RAWPICTURE

        if (enc->codec_type == AVMEDIA_TYPE_VIDEO && (of->ctx->oformat->flags & AVFMT_RAWPICTURE) && enc->codec->id == AV_CODEC_ID_RAWVIDEO)

            continue;

#endif



        if (enc->codec_type != AVMEDIA_TYPE_VIDEO && enc->codec_type != AVMEDIA_TYPE_AUDIO)

            continue;



        avcodec_send_frame(enc, NULL);



        for (;;) {

            const char *desc = NULL;

            AVPacket pkt;

            int pkt_size;



            switch (enc->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                desc   = "audio";

                break;

            case AVMEDIA_TYPE_VIDEO:

                desc   = "video";

                break;

            default:

                av_assert0(0);

            }



                av_init_packet(&pkt);

                pkt.data = NULL;

                pkt.size = 0;



                update_benchmark(NULL);

                ret = avcodec_receive_packet(enc, &pkt);

                update_benchmark("flush_%s %d.%d", desc, ost->file_index, ost->index);

                if (ret < 0 && ret != AVERROR_EOF) {

                    av_log(NULL, AV_LOG_FATAL, "%s encoding failed: %s\n",

                           desc,

                           av_err2str(ret));

                    exit_program(1);

                }

                if (ost->logfile && enc->stats_out) {

                    fprintf(ost->logfile, "%s", enc->stats_out);

                }

                if (ret == AVERROR_EOF) {

                    break;

                }

                if (ost->finished & MUXER_FINISHED) {

                    av_packet_unref(&pkt);

                    continue;

                }

                av_packet_rescale_ts(&pkt, enc->time_base, ost->mux_timebase);

                pkt_size = pkt.size;

                output_packet(of, &pkt, ost);

                if (ost->enc_ctx->codec_type == AVMEDIA_TYPE_VIDEO && vstats_filename) {

                    do_video_stats(ost, pkt_size);

                }

        }

    }

}
