static int transcode(OutputFile *output_files, int nb_output_files,

                     InputFile  *input_files,  int nb_input_files)

{

    int ret, i;

    AVFormatContext *is, *os;

    OutputStream *ost;

    InputStream *ist;

    uint8_t *no_packet;

    int no_packet_count = 0;

    int64_t timer_start;

    int key;



    if (!(no_packet = av_mallocz(nb_input_files)))

        exit_program(1);



    ret = transcode_init(output_files, nb_output_files, input_files, nb_input_files);

    if (ret < 0)

        goto fail;



    if (!using_stdin) {

        av_log(NULL, AV_LOG_INFO, "Press [q] to stop, [?] for help\n");

    }



    timer_start = av_gettime();



    for (; received_sigterm == 0;) {

        int file_index, ist_index;

        AVPacket pkt;

        int64_t ipts_min;

        double opts_min;

        int64_t cur_time= av_gettime();



        ipts_min = INT64_MAX;

        opts_min = 1e100;

        /* if 'q' pressed, exits */

        if (!using_stdin) {

            static int64_t last_time;

            if (received_nb_signals)

                break;

            /* read_key() returns 0 on EOF */

            if(cur_time - last_time >= 100000 && !run_as_daemon){

                key =  read_key();

                last_time = cur_time;

            }else

                key = -1;

            if (key == 'q')

                break;

            if (key == '+') av_log_set_level(av_log_get_level()+10);

            if (key == '-') av_log_set_level(av_log_get_level()-10);

            if (key == 's') qp_hist     ^= 1;

            if (key == 'h'){

                if (do_hex_dump){

                    do_hex_dump = do_pkt_dump = 0;

                } else if(do_pkt_dump){

                    do_hex_dump = 1;

                } else

                    do_pkt_dump = 1;

                av_log_set_level(AV_LOG_DEBUG);

            }

#if CONFIG_AVFILTER

            if (key == 'c' || key == 'C'){

                char buf[4096], target[64], command[256], arg[256] = {0};

                double time;

                int k, n = 0;

                fprintf(stderr, "\nEnter command: <target> <time> <command>[ <argument>]\n");

                i = 0;

                while ((k = read_key()) != '\n' && k != '\r' && i < sizeof(buf)-1)

                    if (k > 0)

                        buf[i++] = k;

                buf[i] = 0;

                if (k > 0 &&

                    (n = sscanf(buf, "%63[^ ] %lf %255[^ ] %255[^\n]", target, &time, command, arg)) >= 3) {

                    av_log(NULL, AV_LOG_DEBUG, "Processing command target:%s time:%f command:%s arg:%s",

                           target, time, command, arg);

                    for (i = 0; i < nb_output_streams; i++) {

                        ost = &output_streams[i];

                        if (ost->graph) {

                            if (time < 0) {

                                ret = avfilter_graph_send_command(ost->graph, target, command, arg, buf, sizeof(buf),

                                                                  key == 'c' ? AVFILTER_CMD_FLAG_ONE : 0);

                                fprintf(stderr, "Command reply for stream %d: ret:%d res:%s\n", i, ret, buf);

                            } else {

                                ret = avfilter_graph_queue_command(ost->graph, target, command, arg, 0, time);

                            }

                        }

                    }

                } else {

                    av_log(NULL, AV_LOG_ERROR,

                           "Parse error, at least 3 arguments were expected, "

                           "only %d given in string '%s'\n", n, buf);

                }

            }

#endif

            if (key == 'd' || key == 'D'){

                int debug=0;

                if(key == 'D') {

                    debug = input_streams[0].st->codec->debug<<1;

                    if(!debug) debug = 1;

                    while(debug & (FF_DEBUG_DCT_COEFF|FF_DEBUG_VIS_QP|FF_DEBUG_VIS_MB_TYPE)) //unsupported, would just crash

                        debug += debug;

                }else

                    if(scanf("%d", &debug)!=1)

                        fprintf(stderr,"error parsing debug value\n");

                for(i=0;i<nb_input_streams;i++) {

                    input_streams[i].st->codec->debug = debug;

                }

                for(i=0;i<nb_output_streams;i++) {

                    ost = &output_streams[i];

                    ost->st->codec->debug = debug;

                }

                if(debug) av_log_set_level(AV_LOG_DEBUG);

                fprintf(stderr,"debug=%d\n", debug);

            }

            if (key == '?'){

                fprintf(stderr, "key    function\n"

                                "?      show this help\n"

                                "+      increase verbosity\n"

                                "-      decrease verbosity\n"

                                "c      Send command to filtergraph\n"

                                "D      cycle through available debug modes\n"

                                "h      dump packets/hex press to cycle through the 3 states\n"

                                "q      quit\n"

                                "s      Show QP histogram\n"

                );

            }

        }



        /* select the stream that we must read now by looking at the

           smallest output pts */

        file_index = -1;

        for (i = 0; i < nb_output_streams; i++) {

            OutputFile *of;

            int64_t ipts;

            double  opts;

            ost = &output_streams[i];

            of = &output_files[ost->file_index];

            os = output_files[ost->file_index].ctx;

            ist = &input_streams[ost->source_index];

            if (ost->is_past_recording_time || no_packet[ist->file_index] ||

                (os->pb && avio_tell(os->pb) >= of->limit_filesize))

                continue;

            opts = ost->st->pts.val * av_q2d(ost->st->time_base);

            ipts = ist->pts;

            if (!input_files[ist->file_index].eof_reached) {

                if (ipts < ipts_min) {

                    ipts_min = ipts;

                    if (input_sync)

                        file_index = ist->file_index;

                }

                if (opts < opts_min) {

                    opts_min = opts;

                    if (!input_sync) file_index = ist->file_index;

                }

            }

            if (ost->frame_number >= ost->max_frames) {

                int j;

                for (j = 0; j < of->ctx->nb_streams; j++)

                    output_streams[of->ost_index + j].is_past_recording_time = 1;

                continue;

            }

        }

        /* if none, if is finished */

        if (file_index < 0) {

            if (no_packet_count) {

                no_packet_count = 0;

                memset(no_packet, 0, nb_input_files);

                usleep(10000);

                continue;

            }

            break;

        }



        /* read a frame from it and output it in the fifo */

        is  = input_files[file_index].ctx;

        ret = av_read_frame(is, &pkt);

        if (ret == AVERROR(EAGAIN)) {

            no_packet[file_index] = 1;

            no_packet_count++;

            continue;

        }

        if (ret < 0) {

            input_files[file_index].eof_reached = 1;

            if (opt_shortest)

                break;

            else

                continue;

        }



        no_packet_count = 0;

        memset(no_packet, 0, nb_input_files);



        if (do_pkt_dump) {

            av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &pkt, do_hex_dump,

                             is->streams[pkt.stream_index]);

        }

        /* the following test is needed in case new streams appear

           dynamically in stream : we ignore them */

        if (pkt.stream_index >= input_files[file_index].nb_streams)

            goto discard_packet;

        ist_index = input_files[file_index].ist_index + pkt.stream_index;

        ist = &input_streams[ist_index];

        if (ist->discard)

            goto discard_packet;



        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts += av_rescale_q(input_files[ist->file_index].ts_offset, AV_TIME_BASE_Q, ist->st->time_base);

        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts += av_rescale_q(input_files[ist->file_index].ts_offset, AV_TIME_BASE_Q, ist->st->time_base);



        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts *= ist->ts_scale;

        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts *= ist->ts_scale;



        //fprintf(stderr, "next:%"PRId64" dts:%"PRId64"/%"PRId64" off:%"PRId64" %d\n",

        //        ist->next_dts,

        //        ist->dts, av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q), input_files[ist->file_index].ts_offset,

        //        ist->st->codec->codec_type);

        if (pkt.dts != AV_NOPTS_VALUE && ist->next_dts != AV_NOPTS_VALUE

            && (is->iformat->flags & AVFMT_TS_DISCONT)) {

            int64_t pkt_dts = av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q);

            int64_t delta   = pkt_dts - ist->next_dts;

            if((delta < -1LL*dts_delta_threshold*AV_TIME_BASE ||

                (delta > 1LL*dts_delta_threshold*AV_TIME_BASE &&

                 ist->st->codec->codec_type != AVMEDIA_TYPE_SUBTITLE) ||

                pkt_dts+1<ist->pts)&& !copy_ts){

                input_files[ist->file_index].ts_offset -= delta;

                av_log(NULL, AV_LOG_DEBUG,

                       "timestamp discontinuity %"PRId64", new offset= %"PRId64"\n",

                       delta, input_files[ist->file_index].ts_offset);

                pkt.dts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

                if (pkt.pts != AV_NOPTS_VALUE)

                    pkt.pts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

            }

        }



        // fprintf(stderr,"read #%d.%d size=%d\n", ist->file_index, ist->st->index, pkt.size);

        if (output_packet(ist, output_streams, nb_output_streams, &pkt) < 0) {



            av_log(NULL, AV_LOG_ERROR, "Error while decoding stream #%d:%d\n",

                   ist->file_index, ist->st->index);

            if (exit_on_error)

                exit_program(1);

            av_free_packet(&pkt);

            continue;

        }



    discard_packet:

        av_free_packet(&pkt);



        /* dump report by using the output first video and audio streams */

        print_report(output_files, output_streams, nb_output_streams, 0, timer_start, cur_time);

    }



    /* at the end of stream, we must flush the decoder buffers */

    for (i = 0; i < nb_input_streams; i++) {

        ist = &input_streams[i];

        if (ist->decoding_needed) {

            output_packet(ist, output_streams, nb_output_streams, NULL);

        }

    }

    flush_encoders(output_streams, nb_output_streams);



    term_exit();



    /* write the trailer if needed and close file */

    for (i = 0; i < nb_output_files; i++) {

        os = output_files[i].ctx;

        av_write_trailer(os);

    }



    /* dump report by using the first video and audio streams */

    print_report(output_files, output_streams, nb_output_streams, 1, timer_start, av_gettime());



    /* close each encoder */

    for (i = 0; i < nb_output_streams; i++) {

        ost = &output_streams[i];

        if (ost->encoding_needed) {

            av_freep(&ost->st->codec->stats_in);

            avcodec_close(ost->st->codec);

        }

#if CONFIG_AVFILTER

        avfilter_graph_free(&ost->graph);

#endif

    }



    /* close each decoder */

    for (i = 0; i < nb_input_streams; i++) {

        ist = &input_streams[i];

        if (ist->decoding_needed) {

            avcodec_close(ist->st->codec);

        }

    }



    /* finished ! */

    ret = 0;



 fail:

    av_freep(&no_packet);



    if (output_streams) {

        for (i = 0; i < nb_output_streams; i++) {

            ost = &output_streams[i];

            if (ost) {

                if (ost->stream_copy)

                    av_freep(&ost->st->codec->extradata);

                if (ost->logfile) {

                    fclose(ost->logfile);

                    ost->logfile = NULL;

                }

                av_fifo_free(ost->fifo); /* works even if fifo is not

                                             initialized but set to zero */

                av_freep(&ost->st->codec->subtitle_header);

                av_free(ost->resample_frame.data[0]);

                av_free(ost->forced_kf_pts);

                if (ost->video_resample)

                    sws_freeContext(ost->img_resample_ctx);

                swr_free(&ost->swr);

                av_dict_free(&ost->opts);

            }

        }

    }

    return ret;

}
