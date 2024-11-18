ff_rm_parse_packet (AVFormatContext *s, ByteIOContext *pb,

                    AVStream *st, RMStream *ast, int len, AVPacket *pkt,

                    int *seq, int *flags, int64_t *timestamp)

{

    RMDemuxContext *rm = s->priv_data;



    if (st->codec->codec_type == CODEC_TYPE_VIDEO) {

        rm->current_stream= st->id;

        if(rm_assemble_video_frame(s, pb, rm, ast, pkt, len))

            return -1; //got partial frame

    } else if (st->codec->codec_type == CODEC_TYPE_AUDIO) {

        if ((st->codec->codec_id == CODEC_ID_RA_288) ||

            (st->codec->codec_id == CODEC_ID_COOK) ||

            (st->codec->codec_id == CODEC_ID_ATRAC3) ||

            (st->codec->codec_id == CODEC_ID_SIPR)) {

            int x;

            int sps = ast->sub_packet_size;

            int cfs = ast->coded_framesize;

            int h = ast->sub_packet_h;

            int y = ast->sub_packet_cnt;

            int w = ast->audio_framesize;



            if (*flags & 2)

                y = ast->sub_packet_cnt = 0;

            if (!y)

                ast->audiotimestamp = *timestamp;



            switch(st->codec->codec_id) {

                case CODEC_ID_RA_288:

                    for (x = 0; x < h/2; x++)

                        get_buffer(pb, ast->pkt.data+x*2*w+y*cfs, cfs);

                    break;

                case CODEC_ID_ATRAC3:

                case CODEC_ID_COOK:

                    for (x = 0; x < w/sps; x++)

                        get_buffer(pb, ast->pkt.data+sps*(h*x+((h+1)/2)*(y&1)+(y>>1)), sps);

                    break;

            }



            if (++(ast->sub_packet_cnt) < h)

                return -1;

            else {

                ast->sub_packet_cnt = 0;

                rm->audio_stream_num = st->index;

                rm->audio_pkt_cnt = h * w / st->codec->block_align - 1;

                // Release first audio packet

                av_new_packet(pkt, st->codec->block_align);

                memcpy(pkt->data, ast->pkt.data, st->codec->block_align); //FIXME avoid this

                *timestamp = ast->audiotimestamp;

                *flags = 2; // Mark first packet as keyframe

            }

        } else if (st->codec->codec_id == CODEC_ID_AAC) {

            int x;

            rm->audio_stream_num = st->index;

            ast->sub_packet_cnt = (get_be16(pb) & 0xf0) >> 4;

            if (ast->sub_packet_cnt) {

                for (x = 0; x < ast->sub_packet_cnt; x++)

                    ast->sub_packet_lengths[x] = get_be16(pb);

                // Release first audio packet

                rm->audio_pkt_cnt = ast->sub_packet_cnt - 1;

                av_get_packet(pb, pkt, ast->sub_packet_lengths[0]);

                *flags = 2; // Mark first packet as keyframe

            }

        } else {

            av_get_packet(pb, pkt, len);

            rm_ac3_swap_bytes(st, pkt);

        }

    } else

        av_get_packet(pb, pkt, len);



    if(  (st->discard >= AVDISCARD_NONKEY && !(*flags&2))

       || st->discard >= AVDISCARD_ALL){

        av_free_packet(pkt);

        return -1;

    }



    pkt->stream_index = st->index;



#if 0

    if (st->codec->codec_type == CODEC_TYPE_VIDEO) {

        if(st->codec->codec_id == CODEC_ID_RV20){

            int seq= 128*(pkt->data[2]&0x7F) + (pkt->data[3]>>1);

            av_log(s, AV_LOG_DEBUG, "%d %"PRId64" %d\n", *timestamp, *timestamp*512LL/25, seq);



            seq |= (*timestamp&~0x3FFF);

            if(seq - *timestamp >  0x2000) seq -= 0x4000;

            if(seq - *timestamp < -0x2000) seq += 0x4000;

        }

    }

#endif



    pkt->pts= *timestamp;

    if (*flags & 2)

        pkt->flags |= PKT_FLAG_KEY;



    return st->codec->codec_type == CODEC_TYPE_AUDIO ? rm->audio_pkt_cnt : 0;

}
