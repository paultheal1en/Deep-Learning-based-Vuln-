static int decode_vop_header(MpegEncContext *s, GetBitContext *gb){

    int time_incr, time_increment;



    s->pict_type = get_bits(gb, 2) + I_TYPE;	/* pict type: I = 0 , P = 1 */

    if(s->pict_type==B_TYPE && s->low_delay && s->vol_control_parameters==0 && !(s->flags & CODEC_FLAG_LOW_DELAY)){

        av_log(s->avctx, AV_LOG_ERROR, "low_delay flag set, but shouldnt, clearing it\n");

        s->low_delay=0;

    }

 

    s->partitioned_frame= s->data_partitioning && s->pict_type!=B_TYPE;

    if(s->partitioned_frame)

        s->decode_mb= mpeg4_decode_partitioned_mb;

    else

        s->decode_mb= ff_h263_decode_mb;



    if(s->time_increment_resolution==0){

        s->time_increment_resolution=1;

//        fprintf(stderr, "time_increment_resolution is illegal\n");

    }

    time_incr=0;

    while (get_bits1(gb) != 0) 

        time_incr++;



    check_marker(gb, "before time_increment");

    

    if(s->time_increment_bits==0){

        av_log(s->avctx, AV_LOG_ERROR, "hmm, seems the headers arnt complete, trying to guess time_increment_bits\n");



        for(s->time_increment_bits=1 ;s->time_increment_bits<16; s->time_increment_bits++){

            if(show_bits(gb, s->time_increment_bits+1)&1) break;

        }



        av_log(s->avctx, AV_LOG_ERROR, "my guess is %d bits ;)\n",s->time_increment_bits);

    }

    

    if(IS_3IV1) time_increment= get_bits1(gb); //FIXME investigate further

    else time_increment= get_bits(gb, s->time_increment_bits);

    

//    printf("%d %X\n", s->time_increment_bits, time_increment);

//printf(" type:%d modulo_time_base:%d increment:%d\n", s->pict_type, time_incr, time_increment);

    if(s->pict_type!=B_TYPE){

        s->last_time_base= s->time_base;

        s->time_base+= time_incr;

        s->time= s->time_base*s->time_increment_resolution + time_increment;

        if(s->workaround_bugs&FF_BUG_UMP4){

            if(s->time < s->last_non_b_time){

//                fprintf(stderr, "header is not mpeg4 compatible, broken encoder, trying to workaround\n");

                s->time_base++;

                s->time+= s->time_increment_resolution;

            }

        }

        s->pp_time= s->time - s->last_non_b_time;

        s->last_non_b_time= s->time;

    }else{

        s->time= (s->last_time_base + time_incr)*s->time_increment_resolution + time_increment;

        s->pb_time= s->pp_time - (s->last_non_b_time - s->time);

        if(s->pp_time <=s->pb_time || s->pp_time <= s->pp_time - s->pb_time || s->pp_time<=0){

//            printf("messed up order, seeking?, skiping current b frame\n");

            return FRAME_SKIPED;

        }

        

        if(s->t_frame==0) s->t_frame= s->time - s->last_time_base;

        if(s->t_frame==0) s->t_frame=1; // 1/0 protection

//printf("%Ld %Ld %d %d\n", s->last_non_b_time, s->time, s->pp_time, s->t_frame); fflush(stdout);

        s->pp_field_time= (  ROUNDED_DIV(s->last_non_b_time, s->t_frame) 

                           - ROUNDED_DIV(s->last_non_b_time - s->pp_time, s->t_frame))*2;

        s->pb_field_time= (  ROUNDED_DIV(s->time, s->t_frame) 

                           - ROUNDED_DIV(s->last_non_b_time - s->pp_time, s->t_frame))*2;

    }

    

    s->current_picture_ptr->pts= s->time*1000LL*1000LL / s->time_increment_resolution;

    if(s->avctx->debug&FF_DEBUG_PTS)

        av_log(s->avctx, AV_LOG_DEBUG, "MPEG4 PTS: %f\n", s->current_picture_ptr->pts/(1000.0*1000.0));

    

    check_marker(gb, "before vop_coded");

    

    /* vop coded */

    if (get_bits1(gb) != 1){

        av_log(s->avctx, AV_LOG_ERROR, "vop not coded\n");

        return FRAME_SKIPED;

    }

//printf("time %d %d %d || %Ld %Ld %Ld\n", s->time_increment_bits, s->time_increment_resolution, s->time_base,

//s->time, s->last_non_b_time, s->last_non_b_time - s->pp_time);  

    if (s->shape != BIN_ONLY_SHAPE && ( s->pict_type == P_TYPE

                          || (s->pict_type == S_TYPE && s->vol_sprite_usage==GMC_SPRITE))) {

        /* rounding type for motion estimation */

	s->no_rounding = get_bits1(gb);

    } else {

	s->no_rounding = 0;

    }

//FIXME reduced res stuff



     if (s->shape != RECT_SHAPE) {

         if (s->vol_sprite_usage != 1 || s->pict_type != I_TYPE) {

             int width, height, hor_spat_ref, ver_spat_ref;

 

             width = get_bits(gb, 13);

             skip_bits1(gb);   /* marker */

             height = get_bits(gb, 13);

             skip_bits1(gb);   /* marker */

             hor_spat_ref = get_bits(gb, 13); /* hor_spat_ref */

             skip_bits1(gb);   /* marker */

             ver_spat_ref = get_bits(gb, 13); /* ver_spat_ref */

         }

         skip_bits1(gb); /* change_CR_disable */

 

         if (get_bits1(gb) != 0) {

             skip_bits(gb, 8); /* constant_alpha_value */

         }

     }

//FIXME complexity estimation stuff

     

     if (s->shape != BIN_ONLY_SHAPE) {

         s->intra_dc_threshold= mpeg4_dc_threshold[ get_bits(gb, 3) ];

         if(!s->progressive_sequence){

             s->top_field_first= get_bits1(gb);

             s->alternate_scan= get_bits1(gb);

         }else

             s->alternate_scan= 0;

     }



     if(s->alternate_scan){

         ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , ff_alternate_vertical_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , ff_alternate_vertical_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_h_scantable, ff_alternate_vertical_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_v_scantable, ff_alternate_vertical_scan);

     } else{

         ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , ff_zigzag_direct);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , ff_zigzag_direct);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_h_scantable, ff_alternate_horizontal_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_v_scantable, ff_alternate_vertical_scan);

     }

 

     if(s->pict_type == S_TYPE && (s->vol_sprite_usage==STATIC_SPRITE || s->vol_sprite_usage==GMC_SPRITE)){

         mpeg4_decode_sprite_trajectory(s);

         if(s->sprite_brightness_change) av_log(s->avctx, AV_LOG_ERROR, "sprite_brightness_change not supported\n");

         if(s->vol_sprite_usage==STATIC_SPRITE) av_log(s->avctx, AV_LOG_ERROR, "static sprite not supported\n");

     }



     if (s->shape != BIN_ONLY_SHAPE) {

         s->qscale = get_bits(gb, s->quant_precision);

         if(s->qscale==0){

             av_log(s->avctx, AV_LOG_ERROR, "Error, header damaged or not MPEG4 header (qscale=0)\n");

             return -1; // makes no sense to continue, as there is nothing left from the image then

         }

  

         if (s->pict_type != I_TYPE) {

             s->f_code = get_bits(gb, 3);	/* fcode_for */

             if(s->f_code==0){

                 av_log(s->avctx, AV_LOG_ERROR, "Error, header damaged or not MPEG4 header (f_code=0)\n");

                 return -1; // makes no sense to continue, as the MV decoding will break very quickly

             }

         }else

             s->f_code=1;

     

         if (s->pict_type == B_TYPE) {

             s->b_code = get_bits(gb, 3);

         }else

             s->b_code=1;



         if(s->avctx->debug&FF_DEBUG_PICT_INFO){

             av_log(s->avctx, AV_LOG_DEBUG, "qp:%d fc:%d,%d %s size:%d pro:%d alt:%d top:%d %spel part:%d resync:%d w:%d a:%d rnd:%d vot:%d%s dc:%d\n", 

                 s->qscale, s->f_code, s->b_code, 

                 s->pict_type == I_TYPE ? "I" : (s->pict_type == P_TYPE ? "P" : (s->pict_type == B_TYPE ? "B" : "S")), 

                 gb->size_in_bits,s->progressive_sequence, s->alternate_scan, s->top_field_first, 

                 s->quarter_sample ? "q" : "h", s->data_partitioning, s->resync_marker, s->num_sprite_warping_points,

                 s->sprite_warping_accuracy, 1-s->no_rounding, s->vo_type, s->vol_control_parameters ? " VOLC" : " ", s->intra_dc_threshold); 

         }



         if(!s->scalability){

             if (s->shape!=RECT_SHAPE && s->pict_type!=I_TYPE) {

                 skip_bits1(gb); // vop shape coding type

             }

         }else{

             if(s->enhancement_type){

                 int load_backward_shape= get_bits1(gb);

                 if(load_backward_shape){

                     av_log(s->avctx, AV_LOG_ERROR, "load backward shape isnt supported\n");

                 }

             }

             skip_bits(gb, 2); //ref_select_code

         }

     }

     /* detect buggy encoders which dont set the low_delay flag (divx4/xvid/opendivx)*/

     // note we cannot detect divx5 without b-frames easyly (allthough its buggy too)

     if(s->vo_type==0 && s->vol_control_parameters==0 && s->divx_version==0 && s->picture_number==0){

         av_log(s->avctx, AV_LOG_ERROR, "looks like this file was encoded with (divx4/(old)xvid/opendivx) -> forcing low_delay flag\n");

         s->low_delay=1;

     }



     s->picture_number++; // better than pic number==0 allways ;)



     s->y_dc_scale_table= ff_mpeg4_y_dc_scale_table; //FIXME add short header support 

     s->c_dc_scale_table= ff_mpeg4_c_dc_scale_table;



     if(s->workaround_bugs&FF_BUG_EDGE){

         s->h_edge_pos= s->width;

         s->v_edge_pos= s->height;

     }

     return 0;

}
