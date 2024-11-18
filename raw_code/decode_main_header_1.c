static int decode_main_header(NUTContext *nut){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = &s->pb;

    uint64_t tmp, end;

    unsigned int stream_count;

    int i, j, tmp_stream, tmp_mul, tmp_pts, tmp_size, count, tmp_res;



    end= get_packetheader(nut, bc, 1);

    end += url_ftell(bc) - 4;



    GET_V(tmp              , tmp >=2 && tmp <= 3)

    GET_V(stream_count     , tmp > 0 && tmp <=MAX_STREAMS)



    nut->max_distance = get_v(bc);

    if(nut->max_distance > 65536){

        av_log(s, AV_LOG_DEBUG, "max_distance %d\n", nut->max_distance);

        nut->max_distance= 65536;

    }



    GET_V(nut->time_base_count, tmp>0 && tmp<INT_MAX / sizeof(AVRational))

    nut->time_base= av_malloc(nut->time_base_count * sizeof(AVRational));



    for(i=0; i<nut->time_base_count; i++){

        GET_V(nut->time_base[i].num, tmp>0 && tmp<(1ULL<<31))

        GET_V(nut->time_base[i].den, tmp>0 && tmp<(1ULL<<31))

        if(ff_gcd(nut->time_base[i].num, nut->time_base[i].den) != 1){

            av_log(s, AV_LOG_ERROR, "time base invalid\n");

            return -1;

        }

    }

    tmp_pts=0;

    tmp_mul=1;

    tmp_stream=0;

    for(i=0; i<256;){

        int tmp_flags = get_v(bc);

        int tmp_fields= get_v(bc);

        if(tmp_fields>0) tmp_pts   = get_s(bc);

        if(tmp_fields>1) tmp_mul   = get_v(bc);

        if(tmp_fields>2) tmp_stream= get_v(bc);

        if(tmp_fields>3) tmp_size  = get_v(bc);

        else             tmp_size  = 0;

        if(tmp_fields>4) tmp_res   = get_v(bc);

        else             tmp_res   = 0;

        if(tmp_fields>5) count     = get_v(bc);

        else             count     = tmp_mul - tmp_size;



        while(tmp_fields-- > 6)

           get_v(bc);



        if(count == 0 || i+count > 256){

            av_log(s, AV_LOG_ERROR, "illegal count %d at %d\n", count, i);

            return -1;

        }

        if(tmp_stream >= stream_count){

            av_log(s, AV_LOG_ERROR, "illegal stream number\n");

            return -1;

        }



        for(j=0; j<count; j++,i++){

            if (i == 'N') {

                nut->frame_code[i].flags= FLAG_INVALID;

                j--;

                continue;

            }

            nut->frame_code[i].flags           = tmp_flags ;

            nut->frame_code[i].pts_delta       = tmp_pts   ;

            nut->frame_code[i].stream_id       = tmp_stream;

            nut->frame_code[i].size_mul        = tmp_mul   ;

            nut->frame_code[i].size_lsb        = tmp_size+j;

            nut->frame_code[i].reserved_count  = tmp_res   ;

        }

    }

    assert(nut->frame_code['N'].flags == FLAG_INVALID);



    if(skip_reserved(bc, end) || check_checksum(bc)){

        av_log(s, AV_LOG_ERROR, "Main header checksum mismatch\n");

        return -1;

    }



    nut->stream = av_mallocz(sizeof(StreamContext)*stream_count);

    for(i=0; i<stream_count; i++){

        av_new_stream(s, i);

    }



    return 0;

}
