static int rv10_decode_packet(AVCodecContext *avctx, 

                             uint8_t *buf, int buf_size)

{

    MpegEncContext *s = avctx->priv_data;

    int i, mb_count, mb_pos, left;



    init_get_bits(&s->gb, buf, buf_size*8);

#if 0

    for(i=0; i<buf_size*8 && i<200; i++)

        printf("%d", get_bits1(&s->gb));

    printf("\n");

    return 0;

#endif

    if(s->codec_id ==CODEC_ID_RV10)

        mb_count = rv10_decode_picture_header(s);

    else

        mb_count = rv20_decode_picture_header(s);

    if (mb_count < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "HEADER ERROR\n");

        return -1;

    }

    

    if (s->mb_x >= s->mb_width ||

        s->mb_y >= s->mb_height) {

        av_log(s->avctx, AV_LOG_ERROR, "POS ERROR %d %d\n", s->mb_x, s->mb_y);

        return -1;

    }

    mb_pos = s->mb_y * s->mb_width + s->mb_x;

    left = s->mb_width * s->mb_height - mb_pos;

    if (mb_count > left) {

        av_log(s->avctx, AV_LOG_ERROR, "COUNT ERROR\n");

        return -1;

    }

//if(s->pict_type == P_TYPE) return 0;



    if (s->mb_x == 0 && s->mb_y == 0) {

        if(MPV_frame_start(s, avctx) < 0)

            return -1;

    }



#ifdef DEBUG

    printf("qscale=%d\n", s->qscale);

#endif



    /* default quantization values */

    if(s->codec_id== CODEC_ID_RV10){

        if(s->mb_y==0) s->first_slice_line=1;

    }else{

        s->first_slice_line=1;    

        s->resync_mb_x= s->mb_x;

        s->resync_mb_y= s->mb_y;

    }

    if(s->h263_aic){

        s->y_dc_scale_table= 

        s->c_dc_scale_table= ff_aic_dc_scale_table;

    }else{

        s->y_dc_scale_table=

        s->c_dc_scale_table= ff_mpeg1_dc_scale_table;

    }



    if(s->modified_quant)

        s->chroma_qscale_table= ff_h263_chroma_qscale_table;

        

    ff_set_qscale(s, s->qscale);



    s->rv10_first_dc_coded[0] = 0;

    s->rv10_first_dc_coded[1] = 0;

    s->rv10_first_dc_coded[2] = 0;



    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    ff_init_block_index(s);

    /* decode each macroblock */

    for(i=0;i<mb_count;i++) {

        int ret;

        ff_update_block_index(s);

#ifdef DEBUG

        printf("**mb x=%d y=%d\n", s->mb_x, s->mb_y);

#endif

        

	s->dsp.clear_blocks(s->block[0]);

        s->mv_dir = MV_DIR_FORWARD;

        s->mv_type = MV_TYPE_16X16; 

        ret=ff_h263_decode_mb(s, s->block);



        if (ret == SLICE_ERROR) {

            av_log(s->avctx, AV_LOG_ERROR, "ERROR at MB %d %d\n", s->mb_x, s->mb_y);

            return -1;

        }

        ff_h263_update_motion_val(s);

        MPV_decode_mb(s, s->block);

        if(s->loop_filter)

            ff_h263_loop_filter(s);



        if (++s->mb_x == s->mb_width) {

            s->mb_x = 0;

            s->mb_y++;

            ff_init_block_index(s);

        }

        if(s->mb_x == s->resync_mb_x)

            s->first_slice_line=0;

        if(ret == SLICE_END) break;

    }



    return buf_size;

}
