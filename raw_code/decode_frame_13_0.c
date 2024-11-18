static int decode_frame(NUTContext *nut, AVPacket *pkt, int frame_code, int frame_type){

    AVFormatContext *s= nut->avf;

    StreamContext *stream;

    ByteIOContext *bc = &s->pb;

    int size, flags, size_mul, size_lsb, stream_id;

    int key_frame = 0;

    int64_t pts = 0;

    const int prefix_len= frame_type == 2 ? 8+1 : 1;

    const int64_t frame_start= url_ftell(bc) - prefix_len;



    flags= nut->frame_code[frame_code].flags;

    size_mul= nut->frame_code[frame_code].size_mul;

    size_lsb= nut->frame_code[frame_code].size_lsb;

    stream_id= nut->frame_code[frame_code].stream_id_plus1 - 1;



    if(flags & FLAG_FRAME_TYPE){

        reset(s);

        if(get_packetheader(nut, bc, prefix_len, 0) < 0)

            return -1;

        if(frame_type!=2)

            frame_type= 1;

    }



    if(stream_id==-1)

        stream_id= get_v(bc);

    if(stream_id >= s->nb_streams){

        av_log(s, AV_LOG_ERROR, "illegal stream_id\n");

        return -1;

    }

    stream= &nut->stream[stream_id];



//    av_log(s, AV_LOG_DEBUG, "ft:%d ppts:%d %d %d\n", frame_type, stream->lru_pts_delta[0], stream->lru_pts_delta[1], stream->lru_pts_delta[2]);

    

    if(flags & FLAG_PRED_KEY_FRAME){

        if(flags & FLAG_KEY_FRAME)

            key_frame= !stream->last_key_frame;

        else

            key_frame= stream->last_key_frame;

    }else{

        key_frame= !!(flags & FLAG_KEY_FRAME);

    }



    if(flags & FLAG_PTS){

        if(flags & FLAG_FULL_PTS){

            pts= get_v(bc);

            if(frame_type && key_frame){

                av_add_index_entry(

                    s->streams[stream_id], 

                    frame_start, 

                    pts, 

                    frame_start - nut->stream[stream_id].last_sync_pos,

                    AVINDEX_KEYFRAME);

                nut->stream[stream_id].last_sync_pos= frame_start;

                assert(nut->packet_start == frame_start);

            }

        }else{

            int64_t mask = (1<<stream->msb_timestamp_shift)-1;

            int64_t delta= stream->last_pts - mask/2;

            pts= ((get_v(bc) - delta)&mask) + delta;

        }

    }else{

        pts= stream->last_pts + stream->lru_pts_delta[(flags&12)>>2];

    }

  

    if(size_mul <= size_lsb){

        size= stream->lru_size[size_lsb - size_mul];

    }else{

        if(flags & FLAG_DATA_SIZE)

            size= size_mul*get_v(bc) + size_lsb;

        else

            size= size_lsb;

    }

      

//av_log(s, AV_LOG_DEBUG, "fs:%lld fc:%d ft:%d kf:%d pts:%lld size:%d\n", frame_start, frame_code, frame_type, key_frame, pts, size);



    if(url_ftell(bc) - nut->packet_start + size > nut->written_packet_size){

        av_log(s, AV_LOG_ERROR, "frame size too large\n");

        return -1;

    }

    

    av_new_packet(pkt, size);

    get_buffer(bc, pkt->data, size);

    pkt->stream_index = stream_id;

    if (key_frame)

	pkt->flags |= PKT_FLAG_KEY;

    pkt->pts = pts * AV_TIME_BASE * stream->rate_den / stream->rate_num;



    update(nut, stream_id, frame_start, frame_type, frame_code, key_frame, size, pts);

    

    return 0;

}
