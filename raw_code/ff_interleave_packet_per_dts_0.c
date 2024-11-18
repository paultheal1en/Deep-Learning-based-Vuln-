int ff_interleave_packet_per_dts(AVFormatContext *s, AVPacket *out,

                                 AVPacket *pkt, int flush)

{

    AVPacketList *pktl;

    int stream_count = 0;

    int i;



    if (pkt) {

        ff_interleave_add_packet(s, pkt, interleave_compare_dts);

    }



    if (s->max_interleave_delta > 0 && s->packet_buffer && !flush) {

        AVPacket *top_pkt = &s->packet_buffer->pkt;

        int64_t delta_dts = INT64_MIN;

        int64_t top_dts = av_rescale_q(top_pkt->dts,

                                       s->streams[top_pkt->stream_index]->time_base,

                                       AV_TIME_BASE_Q);



        for (i = 0; i < s->nb_streams; i++) {

            int64_t last_dts;

            const AVPacketList *last = s->streams[i]->last_in_packet_buffer;



            if (!last)

                continue;



            last_dts = av_rescale_q(last->pkt.dts,

                                    s->streams[i]->time_base,

                                    AV_TIME_BASE_Q);

            delta_dts = FFMAX(delta_dts, last_dts - top_dts);

            stream_count++;

        }



        if (delta_dts > s->max_interleave_delta) {

            av_log(s, AV_LOG_DEBUG,

                   "Delay between the first packet and last packet in the "

                   "muxing queue is %"PRId64" > %"PRId64": forcing output\n",

                   delta_dts, s->max_interleave_delta);

            flush = 1;

        }

    } else {

        for (i = 0; i < s->nb_streams; i++)

            stream_count += !!s->streams[i]->last_in_packet_buffer;

    }





    if (stream_count && (s->internal->nb_interleaved_streams == stream_count || flush)) {

        pktl = s->packet_buffer;

        *out = pktl->pkt;



        s->packet_buffer = pktl->next;

        if (!s->packet_buffer)

            s->packet_buffer_end = NULL;



        if (s->streams[out->stream_index]->last_in_packet_buffer == pktl)

            s->streams[out->stream_index]->last_in_packet_buffer = NULL;

        av_freep(&pktl);

        return 1;

    } else {

        av_init_packet(out);

        return 0;

    }

}
