static int nut_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    NUTContext *nut = s->priv_data;

    ByteIOContext *bc = &s->pb;

    int64_t pos;

    int inited_stream_count;



    nut->avf= s;

    

    av_set_pts_info(s, 60, 1, AV_TIME_BASE);



    /* main header */

    pos=0;

    for(;;){

        if (find_startcode(bc, MAIN_STARTCODE, pos)<0){

            av_log(s, AV_LOG_ERROR, "no main startcode found\n");

            return -1;

        }

        pos= url_ftell(bc);

        if(decode_main_header(nut) >= 0)

            break;

    }

    

    

    s->bit_rate = 0;



    nut->stream = av_malloc(sizeof(StreamContext)*nut->stream_count);



    /* stream headers */

    pos=0;

    for(inited_stream_count=0; inited_stream_count < nut->stream_count;){

        if (find_startcode(bc, STREAM_STARTCODE, pos)<0){

            av_log(s, AV_LOG_ERROR, "not all stream headers found\n");

            return -1;

        }

        pos= url_ftell(bc);

        if(decode_stream_header(nut) >= 0)

            inited_stream_count++;

    }



    /* info headers */

    pos=0;

    for(;;){

        uint64_t startcode= find_any_startcode(bc, pos);

        pos= url_ftell(bc);



        if(startcode==0){

            av_log(s, AV_LOG_ERROR, "EOF before video frames\n");

            return -1;

        }else if(startcode == KEYFRAME_STARTCODE){

            url_fseek(bc, -8, SEEK_CUR); //FIXME

            break;

        }else if(startcode != INFO_STARTCODE){

            continue;

        }



        decode_info_header(nut);

    }



    return 0;

}
