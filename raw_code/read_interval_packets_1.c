static int read_interval_packets(WriterContext *w, AVFormatContext *fmt_ctx,

                                 const ReadInterval *interval, int64_t *cur_ts)

{

    AVPacket pkt, pkt1;

    AVFrame *frame = NULL;

    int ret = 0, i = 0, frame_count = 0;

    int64_t start = -INT64_MAX, end = interval->end;

    int has_start = 0, has_end = interval->has_end && !interval->end_is_offset;



    av_init_packet(&pkt);



    av_log(NULL, AV_LOG_VERBOSE, "Processing read interval ");

    log_read_interval(interval, NULL, AV_LOG_VERBOSE);



    if (interval->has_start) {

        int64_t target;

        if (interval->start_is_offset) {

            if (*cur_ts == AV_NOPTS_VALUE) {

                av_log(NULL, AV_LOG_ERROR,

                       "Could not seek to relative position since current "

                       "timestamp is not defined\n");

                ret = AVERROR(EINVAL);



            target = *cur_ts + interval->start;

        } else {

            target = interval->start;




        av_log(NULL, AV_LOG_VERBOSE, "Seeking to read interval start point %s\n",

               av_ts2timestr(target, &AV_TIME_BASE_Q));

        if ((ret = avformat_seek_file(fmt_ctx, -1, -INT64_MAX, target, INT64_MAX, 0)) < 0) {

            av_log(NULL, AV_LOG_ERROR, "Could not seek to position %"PRId64": %s\n",

                   interval->start, av_err2str(ret));






    frame = av_frame_alloc();





    while (!av_read_frame(fmt_ctx, &pkt)) {

        if (selected_streams[pkt.stream_index]) {

            AVRational tb = fmt_ctx->streams[pkt.stream_index]->time_base;



            if (pkt.pts != AV_NOPTS_VALUE)

                *cur_ts = av_rescale_q(pkt.pts, tb, AV_TIME_BASE_Q);



            if (!has_start && *cur_ts != AV_NOPTS_VALUE) {

                start = *cur_ts;

                has_start = 1;




            if (has_start && !has_end && interval->end_is_offset) {

                end = start + interval->end;

                has_end = 1;




            if (interval->end_is_offset && interval->duration_frames) {

                if (frame_count >= interval->end)

                    break;

            } else if (has_end && *cur_ts != AV_NOPTS_VALUE && *cur_ts >= end) {

                break;




            frame_count++;

            if (do_read_packets) {

                if (do_show_packets)

                    show_packet(w, fmt_ctx, &pkt, i++);

                nb_streams_packets[pkt.stream_index]++;


            if (do_read_frames) {

                pkt1 = pkt;

                while (pkt1.size && process_frame(w, fmt_ctx, frame, &pkt1) > 0);



        av_free_packet(&pkt);


    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;

    //Flush remaining frames that are cached in the decoder

    for (i = 0; i < fmt_ctx->nb_streams; i++) {

        pkt.stream_index = i;

        if (do_read_frames)

            while (process_frame(w, fmt_ctx, frame, &pkt) > 0);




end:

    av_frame_free(&frame);

    if (ret < 0) {

        av_log(NULL, AV_LOG_ERROR, "Could not read packets in interval ");

        log_read_interval(interval, NULL, AV_LOG_ERROR);


    return ret;
