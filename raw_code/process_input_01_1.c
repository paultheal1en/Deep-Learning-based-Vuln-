static int process_input(void)

{

    InputFile *ifile;

    AVFormatContext *is;

    InputStream *ist;

    AVPacket pkt;

    int ret, i, j;



    /* select the stream that we must read now */

    ifile = select_input_file();

    /* if none, if is finished */

    if (!ifile) {

        if (got_eagain()) {

            reset_eagain();

            av_usleep(10000);

            return AVERROR(EAGAIN);

        }

        av_log(NULL, AV_LOG_VERBOSE, "No more inputs to read from.\n");

        return AVERROR_EOF;

    }



    is  = ifile->ctx;

    ret = get_input_packet(ifile, &pkt);



    if (ret == AVERROR(EAGAIN)) {

        ifile->eagain = 1;

        return ret;

    }

    if (ret < 0) {

        if (ret != AVERROR_EOF) {

            print_error(is->filename, ret);

            if (exit_on_error)

                exit(1);

        }

        ifile->eof_reached = 1;



        for (i = 0; i < ifile->nb_streams; i++) {

            ist = input_streams[ifile->ist_index + i];

            if (ist->decoding_needed)

                output_packet(ist, NULL);



            /* mark all outputs that don't go through lavfi as finished */

            for (j = 0; j < nb_output_streams; j++) {

                OutputStream *ost = output_streams[j];



                if (ost->source_index == ifile->ist_index + i &&

                    (ost->stream_copy || ost->enc->type == AVMEDIA_TYPE_SUBTITLE))

                    ost->finished= 1;

            }

        }



        return AVERROR(EAGAIN);

    }



    reset_eagain();



    if (do_pkt_dump) {

        av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &pkt, do_hex_dump,

                         is->streams[pkt.stream_index]);

    }

    /* the following test is needed in case new streams appear

       dynamically in stream : we ignore them */

    if (pkt.stream_index >= ifile->nb_streams)

        goto discard_packet;



    ist = input_streams[ifile->ist_index + pkt.stream_index];

    if (ist->discard)

        goto discard_packet;



    if (pkt.dts != AV_NOPTS_VALUE)

        pkt.dts += av_rescale_q(ifile->ts_offset, AV_TIME_BASE_Q, ist->st->time_base);

    if (pkt.pts != AV_NOPTS_VALUE)

        pkt.pts += av_rescale_q(ifile->ts_offset, AV_TIME_BASE_Q, ist->st->time_base);



    if (pkt.pts != AV_NOPTS_VALUE)

        pkt.pts *= ist->ts_scale;

    if (pkt.dts != AV_NOPTS_VALUE)

        pkt.dts *= ist->ts_scale;



    if (pkt.dts != AV_NOPTS_VALUE && ist->next_dts != AV_NOPTS_VALUE &&

        (is->iformat->flags & AVFMT_TS_DISCONT)) {

        int64_t pkt_dts = av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q);

        int64_t delta   = pkt_dts - ist->next_dts;



        if ((FFABS(delta) > 1LL * dts_delta_threshold * AV_TIME_BASE || pkt_dts + 1 < ist->last_dts) && !copy_ts) {

            ifile->ts_offset -= delta;

            av_log(NULL, AV_LOG_DEBUG,

                   "timestamp discontinuity %"PRId64", new offset= %"PRId64"\n",

                   delta, ifile->ts_offset);

            pkt.dts -= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

            if (pkt.pts != AV_NOPTS_VALUE)

                pkt.pts -= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

        }

    }



    ret = output_packet(ist, &pkt);

    if (ret < 0) {

        av_log(NULL, AV_LOG_ERROR, "Error while decoding stream #%d:%d\n",

               ist->file_index, ist->st->index);

        if (exit_on_error)

            exit(1);

    }



discard_packet:

    av_free_packet(&pkt);



    return 0;

}
