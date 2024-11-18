static int slice_decode_thread(AVCodecContext *c, void *arg){

    MpegEncContext *s= *(void**)arg;

    const uint8_t *buf= s->gb.buffer;

    int mb_y= s->start_mb_y;

    const int field_pic= s->picture_structure != PICT_FRAME;



    s->error_count= (3*(s->end_mb_y - s->start_mb_y)*s->mb_width) >> field_pic;



    for(;;){

        uint32_t start_code;

        int ret;



        ret= mpeg_decode_slice((Mpeg1Context*)s, mb_y, &buf, s->gb.buffer_end - buf);

        emms_c();

//av_log(c, AV_LOG_DEBUG, "ret:%d resync:%d/%d mb:%d/%d ts:%d/%d ec:%d\n",

//ret, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, s->start_mb_y, s->end_mb_y, s->error_count);

        if(ret < 0){

            if(s->resync_mb_x>=0 && s->resync_mb_y>=0)

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, AC_ERROR|DC_ERROR|MV_ERROR);

        }else{

            ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, AC_END|DC_END|MV_END);

        }



        if(s->mb_y == s->end_mb_y)

            return 0;



        start_code= -1;

        buf = ff_find_start_code(buf, s->gb.buffer_end, &start_code);

        mb_y= (start_code - SLICE_MIN_START_CODE) << field_pic;

        if (s->picture_structure == PICT_BOTTOM_FIELD)

            mb_y++;

        if(mb_y < 0 || mb_y >= s->end_mb_y)

            return -1;

    }



    return 0; //not reached

}
