static int mpc_read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    AVStream *st = s->streams[stream_index];

    MPCContext *c = s->priv_data;

    AVPacket pkt1, *pkt = &pkt1;

    int ret;

    int index = av_index_search_timestamp(st, timestamp - DELAY_FRAMES, flags);

    uint32_t lastframe;



    /* if found, seek there */

    if (index >= 0){

        c->curframe = st->index_entries[index].pos;

        return 0;

    }

    /* if timestamp is out of bounds, return error */

    if(timestamp < 0 || timestamp >= c->fcount)

        return -1;

    timestamp -= DELAY_FRAMES;

    /* seek to the furthest known position and read packets until

       we reach desired position */

    lastframe = c->curframe;

    if(c->frames_noted) c->curframe = c->frames_noted - 1;

    while(c->curframe < timestamp){

        ret = av_read_frame(s, pkt);

        if (ret < 0){

            c->curframe = lastframe;

            return ret;

        }

        av_free_packet(pkt);

    }

    return 0;

}
