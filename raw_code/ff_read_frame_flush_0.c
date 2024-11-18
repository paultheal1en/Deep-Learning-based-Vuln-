void ff_read_frame_flush(AVFormatContext *s)

{

    AVStream *st;

    int i, j;



    flush_packet_queue(s);



    s->cur_st = NULL;



    /* for each stream, reset read state */

    for(i = 0; i < s->nb_streams; i++) {

        st = s->streams[i];



        if (st->parser) {

            av_parser_close(st->parser);

            st->parser = NULL;

            av_free_packet(&st->cur_pkt);

        }

        st->last_IP_pts = AV_NOPTS_VALUE;

        st->cur_dts = AV_NOPTS_VALUE; /* we set the current DTS to an unspecified origin */

        st->reference_dts = AV_NOPTS_VALUE;

        /* fail safe */

        st->cur_ptr = NULL;

        st->cur_len = 0;



        st->probe_packets = MAX_PROBE_PACKETS;



        for(j=0; j<MAX_REORDER_DELAY+1; j++)

            st->pts_buffer[j]= AV_NOPTS_VALUE;

    }

}
