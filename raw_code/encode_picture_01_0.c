static void encode_picture(MpegEncContext *s, int picture_number)

{

    int mb_x, mb_y, pdif = 0;

    int i;

    int bits;

    MpegEncContext best_s, backup_s;

    uint8_t bit_buf[2][3000];

    uint8_t bit_buf2[2][3000];

    uint8_t bit_buf_tex[2][3000];

    PutBitContext pb[2], pb2[2], tex_pb[2];



    for(i=0; i<2; i++){

        init_put_bits(&pb    [i], bit_buf    [i], 3000, NULL, NULL);

        init_put_bits(&pb2   [i], bit_buf2   [i], 3000, NULL, NULL);

        init_put_bits(&tex_pb[i], bit_buf_tex[i], 3000, NULL, NULL);

    }



    s->picture_number = picture_number;



    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    

    /* Reset the average MB variance */

    s->current_picture.mb_var_sum = 0;

    s->current_picture.mc_mb_var_sum = 0;



#ifdef CONFIG_RISKY

    /* we need to initialize some time vars before we can encode b-frames */

    // RAL: Condition added for MPEG1VIDEO

    if (s->codec_id == CODEC_ID_MPEG1VIDEO || (s->h263_pred && !s->h263_msmpeg4))

        ff_set_mpeg4_time(s, s->picture_number); 

#endif

        

    s->scene_change_score=0;

    

    s->qscale= (int)(s->frame_qscale + 0.5); //FIXME qscale / ... stuff for ME ratedistoration

    

    if(s->msmpeg4_version){

        if(s->pict_type==I_TYPE)

            s->no_rounding=1;

        else if(s->flipflop_rounding)

            s->no_rounding ^= 1;          

    }else if(s->out_format == FMT_H263){

        if(s->pict_type==I_TYPE)

            s->no_rounding=0;

        else if(s->pict_type!=B_TYPE)

            s->no_rounding ^= 1;          

    }

    /* Estimate motion for every MB */

    s->mb_intra=0; //for the rate distoration & bit compare functions

    if(s->pict_type != I_TYPE){

        if(s->pict_type != B_TYPE){

            if((s->avctx->pre_me && s->last_non_b_pict_type==I_TYPE) || s->avctx->pre_me==2){

                s->me.pre_pass=1;

                s->me.dia_size= s->avctx->pre_dia_size;



                for(mb_y=s->mb_height-1; mb_y >=0 ; mb_y--) {

                    for(mb_x=s->mb_width-1; mb_x >=0 ; mb_x--) {

                        s->mb_x = mb_x;

                        s->mb_y = mb_y;

                        ff_pre_estimate_p_frame_motion(s, mb_x, mb_y);

                    }

                }

                s->me.pre_pass=0;

            }

        }



        s->me.dia_size= s->avctx->dia_size;

        for(mb_y=0; mb_y < s->mb_height; mb_y++) {

            s->block_index[0]= s->block_wrap[0]*(mb_y*2 + 1) - 1;

            s->block_index[1]= s->block_wrap[0]*(mb_y*2 + 1);

            s->block_index[2]= s->block_wrap[0]*(mb_y*2 + 2) - 1;

            s->block_index[3]= s->block_wrap[0]*(mb_y*2 + 2);

            for(mb_x=0; mb_x < s->mb_width; mb_x++) {

                s->mb_x = mb_x;

                s->mb_y = mb_y;

                s->block_index[0]+=2;

                s->block_index[1]+=2;

                s->block_index[2]+=2;

                s->block_index[3]+=2;

                

                /* compute motion vector & mb_type and store in context */

                if(s->pict_type==B_TYPE)

                    ff_estimate_b_frame_motion(s, mb_x, mb_y);

                else

                    ff_estimate_p_frame_motion(s, mb_x, mb_y);

            }

        }

    }else /* if(s->pict_type == I_TYPE) */{

        /* I-Frame */

        //FIXME do we need to zero them?

        memset(s->motion_val[0], 0, sizeof(int16_t)*(s->mb_width*2 + 2)*(s->mb_height*2 + 2)*2);

        memset(s->p_mv_table   , 0, sizeof(int16_t)*(s->mb_width+2)*(s->mb_height+2)*2);

        memset(s->mb_type      , MB_TYPE_INTRA, sizeof(uint8_t)*s->mb_width*s->mb_height);

        

        if(!s->fixed_qscale){

            /* finding spatial complexity for I-frame rate control */

            for(mb_y=0; mb_y < s->mb_height; mb_y++) {

                for(mb_x=0; mb_x < s->mb_width; mb_x++) {

                    int xx = mb_x * 16;

                    int yy = mb_y * 16;

                    uint8_t *pix = s->new_picture.data[0] + (yy * s->linesize) + xx;

                    int varc;

		    int sum = s->dsp.pix_sum(pix, s->linesize);

    

		    varc = (s->dsp.pix_norm1(pix, s->linesize) - (((unsigned)(sum*sum))>>8) + 500 + 128)>>8;



                    s->current_picture.mb_var [s->mb_width * mb_y + mb_x] = varc;

                    s->current_picture.mb_mean[s->mb_width * mb_y + mb_x] = (sum+128)>>8;

                    s->current_picture.mb_var_sum    += varc;

                }

            }

        }

    }

    emms_c();



    if(s->scene_change_score > 0 && s->pict_type == P_TYPE){

        s->pict_type= I_TYPE;

        memset(s->mb_type   , MB_TYPE_INTRA, sizeof(uint8_t)*s->mb_width*s->mb_height);

//printf("Scene change detected, encoding as I Frame %d %d\n", s->current_picture.mb_var_sum, s->current_picture.mc_mb_var_sum);

    }



    if(s->pict_type==P_TYPE || s->pict_type==S_TYPE) {

        s->f_code= ff_get_best_fcode(s, s->p_mv_table, MB_TYPE_INTER);

        

        // RAL: Next call moved into that bloc

        ff_fix_long_p_mvs(s);

    }



    // RAL: All this bloc changed

    if(s->pict_type==B_TYPE){

        int a, b;



        a = ff_get_best_fcode(s, s->b_forw_mv_table, MB_TYPE_FORWARD);

        b = ff_get_best_fcode(s, s->b_bidir_forw_mv_table, MB_TYPE_BIDIR);

        s->f_code = FFMAX(a, b);



        a = ff_get_best_fcode(s, s->b_back_mv_table, MB_TYPE_BACKWARD);

        b = ff_get_best_fcode(s, s->b_bidir_back_mv_table, MB_TYPE_BIDIR);

        s->b_code = FFMAX(a, b);



        ff_fix_long_b_mvs(s, s->b_forw_mv_table, s->f_code, MB_TYPE_FORWARD);

        ff_fix_long_b_mvs(s, s->b_back_mv_table, s->b_code, MB_TYPE_BACKWARD);

        ff_fix_long_b_mvs(s, s->b_bidir_forw_mv_table, s->f_code, MB_TYPE_BIDIR);

        ff_fix_long_b_mvs(s, s->b_bidir_back_mv_table, s->b_code, MB_TYPE_BIDIR);

    }

    

    if (s->fixed_qscale) 

        s->frame_qscale = s->current_picture.quality;

    else

        s->frame_qscale = ff_rate_estimate_qscale(s);



    if(s->adaptive_quant){

#ifdef CONFIG_RISKY

        switch(s->codec_id){

        case CODEC_ID_MPEG4:

            ff_clean_mpeg4_qscales(s);

            break;

        case CODEC_ID_H263:

        case CODEC_ID_H263P:

            ff_clean_h263_qscales(s);

            break;

        }

#endif



        s->qscale= s->current_picture.qscale_table[0];

    }else

        s->qscale= (int)(s->frame_qscale + 0.5);

        

    if (s->out_format == FMT_MJPEG) {

        /* for mjpeg, we do include qscale in the matrix */

        s->intra_matrix[0] = ff_mpeg1_default_intra_matrix[0];

        for(i=1;i<64;i++){

            int j= s->idct_permutation[i];



            s->intra_matrix[j] = CLAMP_TO_8BIT((ff_mpeg1_default_intra_matrix[i] * s->qscale) >> 3);

        }

        convert_matrix(s, s->q_intra_matrix, s->q_intra_matrix16, 

                       s->q_intra_matrix16_bias, s->intra_matrix, s->intra_quant_bias, 8, 8);

    }

    

    //FIXME var duplication

    s->current_picture.key_frame= s->pict_type == I_TYPE;

    s->current_picture.pict_type= s->pict_type;



    if(s->current_picture.key_frame)

        s->picture_in_gop_number=0;



    s->last_bits= get_bit_count(&s->pb);

    switch(s->out_format) {

    case FMT_MJPEG:

        mjpeg_picture_header(s);

        break;

#ifdef CONFIG_RISKY

    case FMT_H263:

        if (s->codec_id == CODEC_ID_WMV2) 

            ff_wmv2_encode_picture_header(s, picture_number);

        else if (s->h263_msmpeg4) 

            msmpeg4_encode_picture_header(s, picture_number);

        else if (s->h263_pred)

            mpeg4_encode_picture_header(s, picture_number);

        else if (s->h263_rv10) 

            rv10_encode_picture_header(s, picture_number);

        else

            h263_encode_picture_header(s, picture_number);

        break;

#endif

    case FMT_MPEG1:

        mpeg1_encode_picture_header(s, picture_number);

        break;

    }

    bits= get_bit_count(&s->pb);

    s->header_bits= bits - s->last_bits;

    s->last_bits= bits;

    s->mv_bits=0;

    s->misc_bits=0;

    s->i_tex_bits=0;

    s->p_tex_bits=0;

    s->i_count=0;

    s->f_count=0;

    s->b_count=0;

    s->skip_count=0;



    for(i=0; i<3; i++){

        /* init last dc values */

        /* note: quant matrix value (8) is implied here */

        s->last_dc[i] = 128;

        

        s->current_picture.error[i] = 0;

    }

    s->mb_incr = 1;

    s->last_mv[0][0][0] = 0;

    s->last_mv[0][0][1] = 0;

    s->last_mv[1][0][0] = 0;

    s->last_mv[1][0][1] = 0;

     

    s->last_mv_dir = 0;



#ifdef CONFIG_RISKY

    if (s->codec_id==CODEC_ID_H263 || s->codec_id==CODEC_ID_H263P)

        s->gob_index = ff_h263_get_gob_height(s);



    if(s->codec_id==CODEC_ID_MPEG4 && s->partitioned_frame)

        ff_mpeg4_init_partitions(s);

#endif



    s->resync_mb_x=0;

    s->resync_mb_y=0;

    s->first_slice_line = 1;

    s->ptr_lastgob = s->pb.buf;

    s->ptr_last_mb_line = s->pb.buf;

    for(mb_y=0; mb_y < s->mb_height; mb_y++) {

        s->y_dc_scale= s->y_dc_scale_table[ s->qscale ];

        s->c_dc_scale= s->c_dc_scale_table[ s->qscale ];

        

        s->block_index[0]= s->block_wrap[0]*(mb_y*2 + 1) - 1;

        s->block_index[1]= s->block_wrap[0]*(mb_y*2 + 1);

        s->block_index[2]= s->block_wrap[0]*(mb_y*2 + 2) - 1;

        s->block_index[3]= s->block_wrap[0]*(mb_y*2 + 2);

        s->block_index[4]= s->block_wrap[4]*(mb_y + 1)                    + s->block_wrap[0]*(s->mb_height*2 + 2);

        s->block_index[5]= s->block_wrap[4]*(mb_y + 1 + s->mb_height + 2) + s->block_wrap[0]*(s->mb_height*2 + 2);

        for(mb_x=0; mb_x < s->mb_width; mb_x++) {

            int mb_type= s->mb_type[mb_y * s->mb_width + mb_x];

            const int xy= (mb_y+1) * (s->mb_width+2) + mb_x + 1;

//            int d;

            int dmin=10000000;



            s->mb_x = mb_x;

            s->mb_y = mb_y;

            s->block_index[0]+=2;

            s->block_index[1]+=2;

            s->block_index[2]+=2;

            s->block_index[3]+=2;

            s->block_index[4]++;

            s->block_index[5]++;



            /* write gob / video packet header  */

#ifdef CONFIG_RISKY

            if(s->rtp_mode){

                int current_packet_size, is_gob_start;

                

                current_packet_size= pbBufPtr(&s->pb) - s->ptr_lastgob;

                is_gob_start=0;

                

                if(s->codec_id==CODEC_ID_MPEG4){

                    if(current_packet_size + s->mb_line_avgsize/s->mb_width >= s->rtp_payload_size

                       && s->mb_y + s->mb_x>0){



                        if(s->partitioned_frame){

                            ff_mpeg4_merge_partitions(s);

                            ff_mpeg4_init_partitions(s);

                        }

                        ff_mpeg4_encode_video_packet_header(s);



                        if(s->flags&CODEC_FLAG_PASS1){

                            int bits= get_bit_count(&s->pb);

                            s->misc_bits+= bits - s->last_bits;

                            s->last_bits= bits;

                        }

                        ff_mpeg4_clean_buffers(s);

                        is_gob_start=1;

                    }

                }else{

                    if(current_packet_size + s->mb_line_avgsize*s->gob_index >= s->rtp_payload_size

                       && s->mb_x==0 && s->mb_y>0 && s->mb_y%s->gob_index==0){

                       

                        h263_encode_gob_header(s, mb_y);                       

                        is_gob_start=1;

                    }

                }



                if(is_gob_start){

                    s->ptr_lastgob = pbBufPtr(&s->pb);

                    s->first_slice_line=1;

                    s->resync_mb_x=mb_x;

                    s->resync_mb_y=mb_y;

                }

            }

#endif



            if(  (s->resync_mb_x   == s->mb_x)

               && s->resync_mb_y+1 == s->mb_y){

                s->first_slice_line=0; 

            }



            if(mb_type & (mb_type-1)){ // more than 1 MB type possible

                int next_block=0;

                int pb_bits_count, pb2_bits_count, tex_pb_bits_count;



                copy_context_before_encode(&backup_s, s, -1);

                backup_s.pb= s->pb;

                best_s.data_partitioning= s->data_partitioning;

                best_s.partitioned_frame= s->partitioned_frame;

                if(s->data_partitioning){

                    backup_s.pb2= s->pb2;

                    backup_s.tex_pb= s->tex_pb;

                }



                if(mb_type&MB_TYPE_INTER){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->p_mv_table[xy][0];

                    s->mv[0][0][1] = s->p_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_INTER, pb, pb2, tex_pb, 

                                 &dmin, &next_block, s->mv[0][0][0], s->mv[0][0][1]);

                }

                if(mb_type&MB_TYPE_INTER4V){                 

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_8X8;

                    s->mb_intra= 0;

                    for(i=0; i<4; i++){

                        s->mv[0][i][0] = s->motion_val[s->block_index[i]][0];

                        s->mv[0][i][1] = s->motion_val[s->block_index[i]][1];

                    }

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_INTER4V, pb, pb2, tex_pb, 

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&MB_TYPE_FORWARD){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_forw_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_FORWARD, pb, pb2, tex_pb, 

                                 &dmin, &next_block, s->mv[0][0][0], s->mv[0][0][1]);

                }

                if(mb_type&MB_TYPE_BACKWARD){

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[1][0][0] = s->b_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_back_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_BACKWARD, pb, pb2, tex_pb, 

                                 &dmin, &next_block, s->mv[1][0][0], s->mv[1][0][1]);

                }

                if(mb_type&MB_TYPE_BIDIR){

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_bidir_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_bidir_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_bidir_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_bidir_back_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_BIDIR, pb, pb2, tex_pb, 

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&MB_TYPE_DIRECT){

                    int mx= s->b_direct_mv_table[xy][0];

                    int my= s->b_direct_mv_table[xy][1];

                    

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

                    s->mb_intra= 0;

#ifdef CONFIG_RISKY

                    ff_mpeg4_set_direct_mv(s, mx, my);

#endif

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_DIRECT, pb, pb2, tex_pb, 

                                 &dmin, &next_block, mx, my);

                }

                if(mb_type&MB_TYPE_INTRA){

                    s->mv_dir = 0;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 1;

                    s->mv[0][0][0] = 0;

                    s->mv[0][0][1] = 0;

                    encode_mb_hq(s, &backup_s, &best_s, MB_TYPE_INTRA, pb, pb2, tex_pb, 

                                 &dmin, &next_block, 0, 0);

                    /* force cleaning of ac/dc pred stuff if needed ... */

                    if(s->h263_pred || s->h263_aic)

                        s->mbintra_table[mb_x + mb_y*s->mb_width]=1;

                }

                copy_context_after_encode(s, &best_s, -1);

                

                pb_bits_count= get_bit_count(&s->pb);

                flush_put_bits(&s->pb);

                ff_copy_bits(&backup_s.pb, bit_buf[next_block^1], pb_bits_count);

                s->pb= backup_s.pb;

                

                if(s->data_partitioning){

                    pb2_bits_count= get_bit_count(&s->pb2);

                    flush_put_bits(&s->pb2);

                    ff_copy_bits(&backup_s.pb2, bit_buf2[next_block^1], pb2_bits_count);

                    s->pb2= backup_s.pb2;

                    

                    tex_pb_bits_count= get_bit_count(&s->tex_pb);

                    flush_put_bits(&s->tex_pb);

                    ff_copy_bits(&backup_s.tex_pb, bit_buf_tex[next_block^1], tex_pb_bits_count);

                    s->tex_pb= backup_s.tex_pb;

                }

                s->last_bits= get_bit_count(&s->pb);

            } else {

                int motion_x, motion_y;

                int intra_score;

                int inter_score= s->current_picture.mb_cmp_score[mb_x + mb_y*s->mb_width];

                

              if(!(s->flags&CODEC_FLAG_HQ) && s->pict_type==P_TYPE){

                /* get luma score */

                if((s->avctx->mb_cmp&0xFF)==FF_CMP_SSE){

                    intra_score= (s->current_picture.mb_var[mb_x + mb_y*s->mb_width]<<8) - 500; //FIXME dont scale it down so we dont have to fix it

                }else{

                    uint8_t *dest_y;



                    int mean= s->current_picture.mb_mean[mb_x + mb_y*s->mb_width]; //FIXME

                    mean*= 0x01010101;

                    

                    dest_y  = s->new_picture.data[0] + (mb_y * 16 * s->linesize    ) + mb_x * 16;

                

                    for(i=0; i<16; i++){

                        *(uint32_t*)(&s->me.scratchpad[i*s->linesize+ 0]) = mean;

                        *(uint32_t*)(&s->me.scratchpad[i*s->linesize+ 4]) = mean;

                        *(uint32_t*)(&s->me.scratchpad[i*s->linesize+ 8]) = mean;

                        *(uint32_t*)(&s->me.scratchpad[i*s->linesize+12]) = mean;

                    }



                    s->mb_intra=1;

                    intra_score= s->dsp.mb_cmp[0](s, s->me.scratchpad, dest_y, s->linesize);

                                        

/*                    printf("intra:%7d inter:%7d var:%7d mc_var.%7d\n", intra_score>>8, inter_score>>8, 

                        s->current_picture.mb_var[mb_x + mb_y*s->mb_width],

                        s->current_picture.mc_mb_var[mb_x + mb_y*s->mb_width]);*/

                }

                

                /* get chroma score */

                if(s->avctx->mb_cmp&FF_CMP_CHROMA){

                    int i;

                    

                    s->mb_intra=1;

                    for(i=1; i<3; i++){

                        uint8_t *dest_c;

                        int mean;

                        

                        if(s->out_format == FMT_H263){

                            mean= (s->dc_val[i][mb_x + (mb_y+1)*(s->mb_width+2)] + 4)>>3; //FIXME not exact but simple ;)

                        }else{

                            mean= (s->last_dc[i] + 4)>>3;

                        }

                        dest_c = s->new_picture.data[i] + (mb_y * 8  * (s->uvlinesize)) + mb_x * 8;

                        

                        mean*= 0x01010101;

                        for(i=0; i<8; i++){

                            *(uint32_t*)(&s->me.scratchpad[i*s->uvlinesize+ 0]) = mean;

                            *(uint32_t*)(&s->me.scratchpad[i*s->uvlinesize+ 4]) = mean;

                        }

                        

                        intra_score+= s->dsp.mb_cmp[1](s, s->me.scratchpad, dest_c, s->uvlinesize);

                    }                

                }



                /* bias */

                switch(s->avctx->mb_cmp&0xFF){

                default:

                case FF_CMP_SAD:

                    intra_score+= 32*s->qscale;

                    break;

                case FF_CMP_SSE:

                    intra_score+= 24*s->qscale*s->qscale;

                    break;

                case FF_CMP_SATD:

                    intra_score+= 96*s->qscale;

                    break;

                case FF_CMP_DCT:

                    intra_score+= 48*s->qscale;

                    break;

                case FF_CMP_BIT:

                    intra_score+= 16;

                    break;

                case FF_CMP_PSNR:

                case FF_CMP_RD:

                    intra_score+= (s->qscale*s->qscale*109*8 + 64)>>7;

                    break;

                }



                if(intra_score < inter_score)

                    mb_type= MB_TYPE_INTRA;

              }  

                

                s->mv_type=MV_TYPE_16X16;

                // only one MB-Type possible

                

                switch(mb_type){

                case MB_TYPE_INTRA:

                    s->mv_dir = 0;

                    s->mb_intra= 1;

                    motion_x= s->mv[0][0][0] = 0;

                    motion_y= s->mv[0][0][1] = 0;

                    break;

                case MB_TYPE_INTER:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[0][0][0] = s->p_mv_table[xy][0];

                    motion_y= s->mv[0][0][1] = s->p_mv_table[xy][1];

                    break;

                case MB_TYPE_INTER4V:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_8X8;

                    s->mb_intra= 0;

                    for(i=0; i<4; i++){

                        s->mv[0][i][0] = s->motion_val[s->block_index[i]][0];

                        s->mv[0][i][1] = s->motion_val[s->block_index[i]][1];

                    }

                    motion_x= motion_y= 0;

                    break;

                case MB_TYPE_DIRECT:

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

                    s->mb_intra= 0;

                    motion_x=s->b_direct_mv_table[xy][0];

                    motion_y=s->b_direct_mv_table[xy][1];

#ifdef CONFIG_RISKY

                    ff_mpeg4_set_direct_mv(s, motion_x, motion_y);

#endif

                    break;

                case MB_TYPE_BIDIR:

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mb_intra= 0;

                    motion_x=0;

                    motion_y=0;

                    s->mv[0][0][0] = s->b_bidir_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_bidir_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_bidir_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_bidir_back_mv_table[xy][1];

                    break;

                case MB_TYPE_BACKWARD:

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[1][0][0] = s->b_back_mv_table[xy][0];

                    motion_y= s->mv[1][0][1] = s->b_back_mv_table[xy][1];

                    break;

                case MB_TYPE_FORWARD:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[0][0][0] = s->b_forw_mv_table[xy][0];

                    motion_y= s->mv[0][0][1] = s->b_forw_mv_table[xy][1];

//                    printf(" %d %d ", motion_x, motion_y);

                    break;

                default:

                    motion_x=motion_y=0; //gcc warning fix

                    printf("illegal MB type\n");

                }



                encode_mb(s, motion_x, motion_y);



                // RAL: Update last macrobloc type

                s->last_mv_dir = s->mv_dir;

            }



            /* clean the MV table in IPS frames for direct mode in B frames */

            if(s->mb_intra /* && I,P,S_TYPE */){

                s->p_mv_table[xy][0]=0;

                s->p_mv_table[xy][1]=0;

            }



            MPV_decode_mb(s, s->block);

            

            if(s->flags&CODEC_FLAG_PSNR){

                int w= 16;

                int h= 16;



                if(s->mb_x*16 + 16 > s->width ) w= s->width - s->mb_x*16;

                if(s->mb_y*16 + 16 > s->height) h= s->height- s->mb_y*16;



                s->current_picture.error[0] += sse(

                    s,

                    s->new_picture    .data[0] + s->mb_x*16 + s->mb_y*s->linesize*16,

                    s->current_picture.data[0] + s->mb_x*16 + s->mb_y*s->linesize*16,

                    w, h, s->linesize);

                s->current_picture.error[1] += sse(

                    s,

                    s->new_picture    .data[1] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,

                    s->current_picture.data[1] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,

                    w>>1, h>>1, s->uvlinesize);

                s->current_picture.error[2] += sse(

                    s,

                    s->new_picture    .data[2] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,

                    s->current_picture.data[2] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,

                    w>>1, h>>1, s->uvlinesize);

            }

//printf("MB %d %d bits\n", s->mb_x+s->mb_y*s->mb_width, get_bit_count(&s->pb));

        }





        /* Obtain average mb_row size for RTP */

        if (s->rtp_mode) {

            if (mb_y==0)

                s->mb_line_avgsize = pbBufPtr(&s->pb) - s->ptr_last_mb_line;

            else {    

                s->mb_line_avgsize = (s->mb_line_avgsize + pbBufPtr(&s->pb) - s->ptr_last_mb_line) >> 1;

            }

            s->ptr_last_mb_line = pbBufPtr(&s->pb);

        }

    }

    emms_c();



#ifdef CONFIG_RISKY

    if(s->codec_id==CODEC_ID_MPEG4 && s->partitioned_frame)

        ff_mpeg4_merge_partitions(s);



    if (s->msmpeg4_version && s->msmpeg4_version<4 && s->pict_type == I_TYPE)

        msmpeg4_encode_ext_header(s);



    if(s->codec_id==CODEC_ID_MPEG4) 

        ff_mpeg4_stuffing(&s->pb);

#endif



    //if (s->gob_number)

    //    fprintf(stderr,"\nNumber of GOB: %d", s->gob_number);

    

    /* Send the last GOB if RTP */    

    if (s->rtp_mode) {

        flush_put_bits(&s->pb);

        pdif = pbBufPtr(&s->pb) - s->ptr_lastgob;

        /* Call the RTP callback to send the last GOB */

        if (s->rtp_callback)

            s->rtp_callback(s->ptr_lastgob, pdif, s->gob_number);

        s->ptr_lastgob = pbBufPtr(&s->pb);

        //fprintf(stderr,"\nGOB: %2d size: %d (last)", s->gob_number, pdif);

    }

}
