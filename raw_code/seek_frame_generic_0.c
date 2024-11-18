static int seek_frame_generic(AVFormatContext *s,

                                 int stream_index, int64_t timestamp, int flags)

{

    int index;

    int64_t ret;

    AVStream *st;

    AVIndexEntry *ie;



    st = s->streams[stream_index];



    index = av_index_search_timestamp(st, timestamp, flags);



    if(index < 0 && st->nb_index_entries && timestamp < st->index_entries[0].timestamp)

        return -1;



    if(index < 0 || index==st->nb_index_entries-1){

        AVPacket pkt;

        int nonkey=0;



        if(st->nb_index_entries){

            assert(st->index_entries);

            ie= &st->index_entries[st->nb_index_entries-1];

            if ((ret = avio_seek(s->pb, ie->pos, SEEK_SET)) < 0)

                return ret;

            ff_update_cur_dts(s, st, ie->timestamp);

        }else{

            if ((ret = avio_seek(s->pb, s->data_offset, SEEK_SET)) < 0)

                return ret;

        }

        for (;;) {

            int read_status;

            do{

                read_status = av_read_frame(s, &pkt);

            } while (read_status == AVERROR(EAGAIN));

            if (read_status < 0)

                break;

            av_free_packet(&pkt);

            if(stream_index == pkt.stream_index && pkt.dts > timestamp){

                if(pkt.flags & AV_PKT_FLAG_KEY)

                    break;

                if(nonkey++ > 1000){

                    av_log(s, AV_LOG_ERROR,"seek_frame_generic failed as this stream seems to contain no keyframes after the target timestamp, %d non keyframes found\n", nonkey);

                    break;

                }

            }

        }

        index = av_index_search_timestamp(st, timestamp, flags);

    }

    if (index < 0)

        return -1;



    ff_read_frame_flush(s);

    AV_NOWARN_DEPRECATED(

    if (s->iformat->read_seek){

        if(s->iformat->read_seek(s, stream_index, timestamp, flags) >= 0)

            return 0;

    }

    )

    ie = &st->index_entries[index];

    if ((ret = avio_seek(s->pb, ie->pos, SEEK_SET)) < 0)

        return ret;

    ff_update_cur_dts(s, st, ie->timestamp);



    return 0;

}
