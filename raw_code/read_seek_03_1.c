static int read_seek(AVFormatContext *s, int stream_index,

                     int64_t timestamp, int flags)

{

    AVStream *st = s->streams[0];

    CaffContext *caf = s->priv_data;

    int64_t pos;



    timestamp = FFMAX(timestamp, 0);



    if (caf->frames_per_packet > 0 && caf->bytes_per_packet > 0) {

        /* calculate new byte position based on target frame position */

        pos = caf->bytes_per_packet * timestamp / caf->frames_per_packet;

        if (caf->data_size > 0)

            pos = FFMIN(pos, caf->data_size);

        caf->packet_cnt = pos / caf->bytes_per_packet;

        caf->frame_cnt  = caf->frames_per_packet * caf->packet_cnt;

    } else if (st->nb_index_entries) {

        caf->packet_cnt = av_index_search_timestamp(st, timestamp, flags);

        caf->frame_cnt  = st->index_entries[caf->packet_cnt].timestamp;

        pos             = st->index_entries[caf->packet_cnt].pos;

    } else {

        return -1;

    }



    avio_seek(s->pb, pos + caf->data_start, SEEK_SET);

    return 0;

}
