static int nut_write_packet(AVFormatContext *s, int stream_index, 

			    const uint8_t *buf, int size, int64_t pts)

{

    NUTContext *nut = s->priv_data;

    StreamContext *stream= &nut->stream[stream_index];

    ByteIOContext *bc = &s->pb;

    int key_frame = 0, full_pts=0;

    AVCodecContext *enc;

    int64_t lsb_pts, delta_pts;

    int frame_type, best_length, frame_code, flags, i, size_mul, size_lsb;

    const int64_t frame_start= url_ftell(bc);



    if (stream_index > s->nb_streams)

	return 1;

        

    pts= (av_rescale(pts, stream->rate_num, stream->rate_den) + AV_TIME_BASE/2) / AV_TIME_BASE;



    enc = &s->streams[stream_index]->codec;

    key_frame = enc->coded_frame->key_frame;

    delta_pts= pts - stream->last_pts;

    

    frame_type=0;

    if(frame_start + size + 20 - FFMAX(nut->last_frame_start[1], nut->last_frame_start[2]) > MAX_TYPE1_DISTANCE)

        frame_type=1;

    if(key_frame){

        if(frame_type==1 && frame_start + size - nut->last_frame_start[2] > MAX_TYPE2_DISTANCE)

            frame_type=2;

        if(!stream->last_key_frame)

            frame_type=2;

    }



    if(frame_type>0){

        update_packetheader(nut, bc, 0);

        reset(s);

        full_pts=1;

    }

    //FIXME ensure that the timestamp can be represented by either delta or lsb or full_pts=1



    lsb_pts = pts & ((1 << stream->msb_timestamp_shift)-1);



    best_length=INT_MAX;

    frame_code= -1;

    for(i=0; i<256; i++){

        int stream_id_plus1= nut->frame_code[i].stream_id_plus1;

        int fc_key_frame= stream->last_key_frame;

        int length=0;

        size_mul= nut->frame_code[i].size_mul;

        size_lsb= nut->frame_code[i].size_lsb;

        flags= nut->frame_code[i].flags;



        if(stream_id_plus1 == 0) length+= get_length(stream_index);

        else if(stream_id_plus1 - 1 != stream_index)

            continue;

        if(flags & FLAG_PRED_KEY_FRAME){

            if(flags & FLAG_KEY_FRAME)

                fc_key_frame= !fc_key_frame;

        }else{

            fc_key_frame= !!(flags & FLAG_KEY_FRAME);

        }

        assert(key_frame==0 || key_frame==1);

        if(fc_key_frame != key_frame)

            continue;



        if((!!(flags & FLAG_FRAME_TYPE)) != (frame_type > 0))

            continue;

        

        if(size_mul <= size_lsb){

            int p= stream->lru_size[size_lsb - size_mul];

            if(p != size)

                continue;

        }else{

            if(size % size_mul != size_lsb)

                continue;

            if(flags & FLAG_DATA_SIZE)

                length += get_length(size / size_mul);

            else if(size/size_mul)

                continue;

        }



        if(full_pts != ((flags & FLAG_PTS) && (flags & FLAG_FULL_PTS)))

            continue;

            

        if(flags&FLAG_PTS){

            if(flags&FLAG_FULL_PTS){

                length += get_length(pts);

            }else{

                length += get_length(lsb_pts);

            }

        }else{

            int delta= stream->lru_pts_delta[(flags & 12)>>2];

            if(delta != pts - stream->last_pts)

                continue;

            assert(frame_type == 0);

        }



        if(length < best_length){

            best_length= length;

            frame_code=i;

        }

//    av_log(s, AV_LOG_DEBUG, "%d %d %d %d %d %d %d %d %d %d\n", key_frame, frame_type, full_pts, size, stream_index, flags, size_mul, size_lsb, stream_id_plus1, length);

    }



    assert(frame_code != -1);

    flags= nut->frame_code[frame_code].flags;

    size_mul= nut->frame_code[frame_code].size_mul;

    size_lsb= nut->frame_code[frame_code].size_lsb;

#if 0

    best_length /= 7;

    best_length ++; //frame_code

    if(frame_type>0){

        best_length += 4; //packet header

        if(frame_type>1)

            best_length += 8; // startcode

    }

    av_log(s, AV_LOG_DEBUG, "kf:%d ft:%d pt:%d fc:%2X len:%2d size:%d stream:%d flag:%d mul:%d lsb:%d s+1:%d pts_delta:%d\n", key_frame, frame_type, full_pts ? 2 : ((flags & FLAG_PTS) ? 1 : 0), frame_code, best_length, size, stream_index, flags, size_mul, size_lsb, nut->frame_code[frame_code].stream_id_plus1,(int)(pts - stream->last_pts));

#endif



    if (frame_type==2)

	put_be64(bc, KEYFRAME_STARTCODE);

    put_byte(bc, frame_code);



    if(frame_type>0)

        put_packetheader(nut, bc, FFMAX(size+20, MAX_TYPE1_DISTANCE));

    if(nut->frame_code[frame_code].stream_id_plus1 == 0)

        put_v(bc, stream_index);

    if (flags & FLAG_PTS){

        if (flags & FLAG_FULL_PTS)

            put_v(bc, pts);

        else

            put_v(bc, lsb_pts);

    }

    if(flags & FLAG_DATA_SIZE)

        put_v(bc, size / size_mul);

    if(size > MAX_TYPE1_DISTANCE){

        assert(frame_type > 0);

        update_packetheader(nut, bc, size);

    }

    

    put_buffer(bc, buf, size);



    update(nut, stream_index, frame_start, frame_type, frame_code, key_frame, size, pts);

    

    return 0;

}
