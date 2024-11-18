static int decode_slice(struct AVCodecContext *avctx, void *arg){

    H264Context *h = *(void**)arg;

    MpegEncContext * const s = &h->s;

    const int part_mask= s->partitioned_frame ? (AC_END|AC_ERROR) : 0x7F;

    int lf_x_start = s->mb_x;



    s->mb_skip_run= -1;



    h->is_complex = FRAME_MBAFF || s->picture_structure != PICT_FRAME || s->codec_id != CODEC_ID_H264 ||

                    (CONFIG_GRAY && (s->flags&CODEC_FLAG_GRAY));



    if( h->pps.cabac ) {

        /* realign */

        align_get_bits( &s->gb );



        /* init cabac */

        ff_init_cabac_states( &h->cabac);

        ff_init_cabac_decoder( &h->cabac,

                               s->gb.buffer + get_bits_count(&s->gb)/8,

                               (get_bits_left(&s->gb) + 7)/8);



        ff_h264_init_cabac_states(h);



        for(;;){

//START_TIMER

            int ret = ff_h264_decode_mb_cabac(h);

            int eos;

//STOP_TIMER("decode_mb_cabac")



            if(ret>=0) ff_h264_hl_decode_mb(h);



            if( ret >= 0 && FRAME_MBAFF ) { //FIXME optimal? or let mb_decode decode 16x32 ?

                s->mb_y++;



                ret = ff_h264_decode_mb_cabac(h);



                if(ret>=0) ff_h264_hl_decode_mb(h);

                s->mb_y--;

            }

            eos = get_cabac_terminate( &h->cabac );



            if((s->workaround_bugs & FF_BUG_TRUNCATED) && h->cabac.bytestream > h->cabac.bytestream_end + 2){

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);

                if (s->mb_x >= lf_x_start) loop_filter(h, lf_x_start, s->mb_x + 1);

                return 0;

            }

            if( ret < 0 || h->cabac.bytestream > h->cabac.bytestream_end + 2) {

                av_log(h->s.avctx, AV_LOG_ERROR, "error while decoding MB %d %d, bytestream (%td)\n", s->mb_x, s->mb_y, h->cabac.bytestream_end - h->cabac.bytestream);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);

                return -1;

            }



            if( ++s->mb_x >= s->mb_width ) {

                loop_filter(h, lf_x_start, s->mb_x);

                s->mb_x = lf_x_start = 0;

                decode_finish_row(h);

                ++s->mb_y;

                if(FIELD_OR_MBAFF_PICTURE) {

                    ++s->mb_y;

                    if(FRAME_MBAFF && s->mb_y < s->mb_height)

                        predict_field_decoding_flag(h);

                }

            }



            if( eos || s->mb_y >= s->mb_height ) {

                tprintf(s->avctx, "slice end %d %d\n", get_bits_count(&s->gb), s->gb.size_in_bits);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);

                if (s->mb_x > lf_x_start) loop_filter(h, lf_x_start, s->mb_x);

                return 0;

            }

        }



    } else {

        for(;;){

            int ret = ff_h264_decode_mb_cavlc(h);



            if(ret>=0) ff_h264_hl_decode_mb(h);



            if(ret>=0 && FRAME_MBAFF){ //FIXME optimal? or let mb_decode decode 16x32 ?

                s->mb_y++;

                ret = ff_h264_decode_mb_cavlc(h);



                if(ret>=0) ff_h264_hl_decode_mb(h);

                s->mb_y--;

            }



            if(ret<0){

                av_log(h->s.avctx, AV_LOG_ERROR, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);

                return -1;

            }



            if(++s->mb_x >= s->mb_width){

                loop_filter(h, lf_x_start, s->mb_x);

                s->mb_x = lf_x_start = 0;

                decode_finish_row(h);

                ++s->mb_y;

                if(FIELD_OR_MBAFF_PICTURE) {

                    ++s->mb_y;

                    if(FRAME_MBAFF && s->mb_y < s->mb_height)

                        predict_field_decoding_flag(h);

                }

                if(s->mb_y >= s->mb_height){

                    tprintf(s->avctx, "slice end %d %d\n", get_bits_count(&s->gb), s->gb.size_in_bits);



                    if(get_bits_count(&s->gb) == s->gb.size_in_bits ) {

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return 0;

                    }else{

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return -1;

                    }

                }

            }



            if(get_bits_count(&s->gb) >= s->gb.size_in_bits && s->mb_skip_run<=0){

                tprintf(s->avctx, "slice end %d %d\n", get_bits_count(&s->gb), s->gb.size_in_bits);

                if(get_bits_count(&s->gb) == s->gb.size_in_bits ){

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);

                    if (s->mb_x > lf_x_start) loop_filter(h, lf_x_start, s->mb_x);



                    return 0;

                }else{

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                    return -1;

                }

            }

        }

    }



#if 0

    for(;s->mb_y < s->mb_height; s->mb_y++){

        for(;s->mb_x < s->mb_width; s->mb_x++){

            int ret= decode_mb(h);



            ff_h264_hl_decode_mb(h);



            if(ret<0){

                av_log(s->avctx, AV_LOG_ERROR, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                return -1;

            }



            if(++s->mb_x >= s->mb_width){

                s->mb_x=0;

                if(++s->mb_y >= s->mb_height){

                    if(get_bits_count(s->gb) == s->gb.size_in_bits){

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return 0;

                    }else{

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return -1;

                    }

                }

            }



            if(get_bits_count(s->?gb) >= s->gb?.size_in_bits){

                if(get_bits_count(s->gb) == s->gb.size_in_bits){

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                    return 0;

                }else{

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                    return -1;

                }

            }

        }

        s->mb_x=0;

        ff_draw_horiz_band(s, 16*s->mb_y, 16);

    }

#endif

    return -1; //not reached

}
