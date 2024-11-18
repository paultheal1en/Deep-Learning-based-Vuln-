static int decode_syncpoint(NUTContext *nut, int64_t *ts, int64_t *back_ptr){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = &s->pb;

    int64_t end, tmp;

    AVRational time_base;



    nut->last_syncpoint_pos= url_ftell(bc)-8;



    end= get_packetheader(nut, bc, 1);

    end += url_ftell(bc);



    tmp= get_v(bc);

    *back_ptr= nut->last_syncpoint_pos - 16*get_v(bc);

    if(*back_ptr < 0)

        return -1;



    ff_nut_reset_ts(nut, nut->time_base[tmp % nut->time_base_count], tmp);



    if(skip_reserved(bc, end) || get_checksum(bc)){

        av_log(s, AV_LOG_ERROR, "sync point checksum mismatch\n");

        return -1;

    }



    *ts= tmp / s->nb_streams * av_q2d(nut->time_base[tmp % s->nb_streams])*AV_TIME_BASE;

    add_sp(nut, nut->last_syncpoint_pos, *back_ptr, *ts);



    return 0;

}
