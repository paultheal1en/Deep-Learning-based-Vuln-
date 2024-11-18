static void do_audio_out(AVFormatContext *s, OutputStream *ost,

                         AVFrame *frame)

{

    AVCodecContext *enc = ost->st->codec;

    AVPacket pkt;

    int got_packet = 0;



    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;

#if 0

    if (!check_recording_time(ost))

        return;

#endif

    if (frame->pts == AV_NOPTS_VALUE || audio_sync_method < 0)

        frame->pts = ost->sync_opts;

    ost->sync_opts = frame->pts + frame->nb_samples;



    av_assert0(pkt.size || !pkt.data);

    update_benchmark(NULL);

    if (avcodec_encode_audio2(enc, &pkt, frame, &got_packet) < 0) {

        av_log(NULL, AV_LOG_FATAL, "Audio encoding failed (avcodec_encode_audio2)\n");

        exit_program(1);

    }

    update_benchmark("encode_audio %d.%d", ost->file_index, ost->index);



    if (got_packet) {

        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts      = av_rescale_q(pkt.pts,      enc->time_base, ost->st->time_base);

        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts      = av_rescale_q(pkt.dts,      enc->time_base, ost->st->time_base);

        if (pkt.duration > 0)

            pkt.duration = av_rescale_q(pkt.duration, enc->time_base, ost->st->time_base);



        if (debug_ts) {

            av_log(NULL, AV_LOG_INFO, "encoder -> type:audio "

                   "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s\n",

                   av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &ost->st->time_base),

                   av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &ost->st->time_base));

        }



        write_frame(s, &pkt, ost);



        audio_size += pkt.size;

        av_free_packet(&pkt);

    }

}
