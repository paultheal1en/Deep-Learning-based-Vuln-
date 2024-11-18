static int nut_write_packet(AVFormatContext *s, AVPacket *pkt){

    NUTContext *nut = s->priv_data;

    StreamContext *nus= &nut->stream[pkt->stream_index];

    AVIOContext *bc = s->pb, *dyn_bc;

    FrameCode *fc;

    int64_t coded_pts;

    int best_length, frame_code, flags, needed_flags, i, header_idx, best_header_idx;

    int key_frame = !!(pkt->flags & AV_PKT_FLAG_KEY);

    int store_sp=0;

    int ret;



    if(pkt->pts < 0)

        return -1;



    if(1LL<<(20+3*nut->header_count) <= avio_tell(bc))

        write_headers(s, bc);



    if(key_frame && !(nus->last_flags & FLAG_KEY))

        store_sp= 1;



    if(pkt->size + 30/*FIXME check*/ + avio_tell(bc) >= nut->last_syncpoint_pos + nut->max_distance)

        store_sp= 1;



//FIXME: Ensure store_sp is 1 in the first place.



    if(store_sp){

        Syncpoint *sp, dummy= {.pos= INT64_MAX};



        ff_nut_reset_ts(nut, *nus->time_base, pkt->dts);

        for(i=0; i<s->nb_streams; i++){

            AVStream *st= s->streams[i];

            int64_t dts_tb = av_rescale_rnd(pkt->dts,

                nus->time_base->num * (int64_t)nut->stream[i].time_base->den,

                nus->time_base->den * (int64_t)nut->stream[i].time_base->num,

                AV_ROUND_DOWN);

            int index= av_index_search_timestamp(st, dts_tb, AVSEEK_FLAG_BACKWARD);

            if(index>=0) dummy.pos= FFMIN(dummy.pos, st->index_entries[index].pos);

        }

        if(dummy.pos == INT64_MAX)

            dummy.pos= 0;

        sp= av_tree_find(nut->syncpoints, &dummy, (void *) ff_nut_sp_pos_cmp,

                         NULL);



        nut->last_syncpoint_pos= avio_tell(bc);

        ret = avio_open_dyn_buf(&dyn_bc);

        if(ret < 0)

            return ret;

        put_tt(nut, nus->time_base, dyn_bc, pkt->dts);

        ff_put_v(dyn_bc, sp ? (nut->last_syncpoint_pos - sp->pos)>>4 : 0);

        put_packet(nut, bc, dyn_bc, 1, SYNCPOINT_STARTCODE);



        ff_nut_add_sp(nut, nut->last_syncpoint_pos, 0/*unused*/, pkt->dts);

    }

    av_assert0(nus->last_pts != AV_NOPTS_VALUE);



    coded_pts = pkt->pts & ((1<<nus->msb_pts_shift)-1);

    if(ff_lsb2full(nus, coded_pts) != pkt->pts)

        coded_pts= pkt->pts + (1<<nus->msb_pts_shift);



    best_header_idx= find_best_header_idx(nut, pkt);



    best_length=INT_MAX;

    frame_code= -1;

    for(i=0; i<256; i++){

        int length= 0;

        FrameCode *fc= &nut->frame_code[i];

        int flags= fc->flags;



        if(flags & FLAG_INVALID)

            continue;

        needed_flags= get_needed_flags(nut, nus, fc, pkt);



        if(flags & FLAG_CODED){

            length++;

            flags = needed_flags;

        }



        if((flags & needed_flags) != needed_flags)

            continue;



        if((flags ^ needed_flags) & FLAG_KEY)

            continue;



        if(flags & FLAG_STREAM_ID)

            length+= ff_get_v_length(pkt->stream_index);



        if(pkt->size % fc->size_mul != fc->size_lsb)

            continue;

        if(flags & FLAG_SIZE_MSB)

            length += ff_get_v_length(pkt->size / fc->size_mul);



        if(flags & FLAG_CHECKSUM)

            length+=4;



        if(flags & FLAG_CODED_PTS)

            length += ff_get_v_length(coded_pts);



        if(   (flags & FLAG_CODED)

           && nut->header_len[best_header_idx] > nut->header_len[fc->header_idx]+1){

            flags |= FLAG_HEADER_IDX;

        }



        if(flags & FLAG_HEADER_IDX){

            length += 1 - nut->header_len[best_header_idx];

        }else{

            length -= nut->header_len[fc->header_idx];

        }



        length*=4;

        length+= !(flags & FLAG_CODED_PTS);

        length+= !(flags & FLAG_CHECKSUM);



        if(length < best_length){

            best_length= length;

            frame_code=i;

        }

    }

    av_assert0(frame_code != -1);

    fc= &nut->frame_code[frame_code];

    flags= fc->flags;

    needed_flags= get_needed_flags(nut, nus, fc, pkt);

    header_idx= fc->header_idx;



    ffio_init_checksum(bc, ff_crc04C11DB7_update, 0);

    avio_w8(bc, frame_code);

    if(flags & FLAG_CODED){

        ff_put_v(bc, (flags^needed_flags) & ~(FLAG_CODED));

        flags = needed_flags;

    }

    if(flags & FLAG_STREAM_ID)  ff_put_v(bc, pkt->stream_index);

    if(flags & FLAG_CODED_PTS)  ff_put_v(bc, coded_pts);

    if(flags & FLAG_SIZE_MSB)   ff_put_v(bc, pkt->size / fc->size_mul);

    if(flags & FLAG_HEADER_IDX) ff_put_v(bc, header_idx= best_header_idx);



    if(flags & FLAG_CHECKSUM)   avio_wl32(bc, ffio_get_checksum(bc));

    else                        ffio_get_checksum(bc);



    avio_write(bc, pkt->data + nut->header_len[header_idx], pkt->size - nut->header_len[header_idx]);

    nus->last_flags= flags;

    nus->last_pts= pkt->pts;



    //FIXME just store one per syncpoint

    if(flags & FLAG_KEY)

        av_add_index_entry(

            s->streams[pkt->stream_index],

            nut->last_syncpoint_pos,

            pkt->pts,

            0,

            0,

            AVINDEX_KEYFRAME);



    return 0;

}
