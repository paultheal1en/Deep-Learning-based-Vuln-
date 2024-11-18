int av_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret, i;

    AVStream *st;



    for(;;){

        AVPacketList *pktl = s->raw_packet_buffer;



        if (pktl) {

            *pkt = pktl->pkt;

            if(s->streams[pkt->stream_index]->codec->codec_id != CODEC_ID_PROBE ||

               !s->streams[pkt->stream_index]->probe_packets){

                s->raw_packet_buffer = pktl->next;

                av_free(pktl);

                return 0;

            }

        }



        av_init_packet(pkt);

        ret= s->iformat->read_packet(s, pkt);

        if (ret < 0) {

            if (!pktl || ret == AVERROR(EAGAIN))

                return ret;

            for (i = 0; i < s->nb_streams; i++)

                s->streams[i]->probe_packets = 0;

            continue;

        }

        st= s->streams[pkt->stream_index];



        switch(st->codec->codec_type){

        case CODEC_TYPE_VIDEO:

            if(s->video_codec_id)   st->codec->codec_id= s->video_codec_id;

            break;

        case CODEC_TYPE_AUDIO:

            if(s->audio_codec_id)   st->codec->codec_id= s->audio_codec_id;

            break;

        case CODEC_TYPE_SUBTITLE:

            if(s->subtitle_codec_id)st->codec->codec_id= s->subtitle_codec_id;

            break;

        }



        if(!pktl && (st->codec->codec_id != CODEC_ID_PROBE ||

                     !st->probe_packets))

            return ret;



        add_to_pktbuf(&s->raw_packet_buffer, pkt, &s->raw_packet_buffer_end);



        if(st->codec->codec_id == CODEC_ID_PROBE){

            AVProbeData *pd = &st->probe_data;



            --st->probe_packets;



            pd->buf = av_realloc(pd->buf, pd->buf_size+pkt->size+AVPROBE_PADDING_SIZE);

            memcpy(pd->buf+pd->buf_size, pkt->data, pkt->size);

            pd->buf_size += pkt->size;

            memset(pd->buf+pd->buf_size, 0, AVPROBE_PADDING_SIZE);



            if(av_log2(pd->buf_size) != av_log2(pd->buf_size - pkt->size)){

                set_codec_from_probe_data(st, pd, 1);

                if(st->codec->codec_id != CODEC_ID_PROBE){

                    pd->buf_size=0;

                    av_freep(&pd->buf);

                }

            }

        }

    }

}
