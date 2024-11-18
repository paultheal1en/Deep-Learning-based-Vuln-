static int64_t mpegts_get_dts(AVFormatContext *s, int stream_index,

                              int64_t *ppos, int64_t pos_limit)

{

    MpegTSContext *ts = s->priv_data;

    int64_t pos;

    int pos47 = ts->pos47_full % ts->raw_packet_size;

    pos = ((*ppos  + ts->raw_packet_size - 1 - pos47) / ts->raw_packet_size) * ts->raw_packet_size + pos47;

    ff_read_frame_flush(s);

    if (avio_seek(s->pb, pos, SEEK_SET) < 0)

        return AV_NOPTS_VALUE;

    while(pos < pos_limit) {

        int ret;

        AVPacket pkt;

        av_init_packet(&pkt);

        ret= av_read_frame(s, &pkt);

        if(ret < 0)

            return AV_NOPTS_VALUE;

        av_free_packet(&pkt);

        if(pkt.dts != AV_NOPTS_VALUE && pkt.pos >= 0){

            ff_reduce_index(s, pkt.stream_index);

            av_add_index_entry(s->streams[pkt.stream_index], pkt.pos, pkt.dts, 0, 0, AVINDEX_KEYFRAME /* FIXME keyframe? */);

            if(pkt.stream_index == stream_index){

                *ppos= pkt.pos;

                return pkt.dts;

            }

        }

        pos = pkt.pos;

    }



    return AV_NOPTS_VALUE;

}
