static int av_seek_frame_generic(AVFormatContext *s,

                                 int stream_index, int64_t timestamp, int flags)

{

    int index;

    AVStream *st;

    AVIndexEntry *ie;



    st = s->streams[stream_index];



    index = av_index_search_timestamp(st, timestamp, flags);



    if(index < 0 || index==st->nb_index_entries-1){

        int i;

        AVPacket pkt;



        if(st->nb_index_entries){

            assert(st->index_entries);

            ie= &st->index_entries[st->nb_index_entries-1];

            url_fseek(s->pb, ie->pos, SEEK_SET);

            av_update_cur_dts(s, st, ie->timestamp);

        }else

            url_fseek(s->pb, 0, SEEK_SET);



        for(i=0;; i++) {

            int ret = av_read_frame(s, &pkt);

            if(ret<0)

                break;

            av_free_packet(&pkt);

            if(stream_index == pkt.stream_index){

                if((pkt.flags & PKT_FLAG_KEY) && pkt.dts > timestamp)

                    break;

            }

        }

        index = av_index_search_timestamp(st, timestamp, flags);

    }

    if (index < 0)

        return -1;



    av_read_frame_flush(s);

    if (s->iformat->read_seek){

        if(s->iformat->read_seek(s, stream_index, timestamp, flags) >= 0)

            return 0;

    }

    ie = &st->index_entries[index];

    url_fseek(s->pb, ie->pos, SEEK_SET);



    av_update_cur_dts(s, st, ie->timestamp);



    return 0;

}
