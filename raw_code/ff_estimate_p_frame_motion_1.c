void ff_estimate_p_frame_motion(MpegEncContext * s,

                                int mb_x, int mb_y)

{

    MotionEstContext * const c= &s->me;

    uint8_t *pix, *ppix;

    int sum, mx, my, dmin;

    int varc;            ///< the variance of the block (sum of squared (p[y][x]-average))

    int vard;            ///< sum of squared differences with the estimated motion vector

    int P[10][2];

    const int shift= 1+s->quarter_sample;

    int mb_type=0;

    Picture * const pic= &s->current_picture;



    init_ref(c, s->new_picture.f.data, s->last_picture.f.data, NULL, 16*mb_x, 16*mb_y, 0);



    assert(s->quarter_sample==0 || s->quarter_sample==1);

    assert(s->linesize == c->stride);

    assert(s->uvlinesize == c->uvstride);



    c->penalty_factor    = get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_cmp);

    c->sub_penalty_factor= get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_sub_cmp);

    c->mb_penalty_factor = get_penalty_factor(s->lambda, s->lambda2, c->avctx->mb_cmp);

    c->current_mv_penalty= c->mv_penalty[s->f_code] + MAX_MV;



    get_limits(s, 16*mb_x, 16*mb_y);

    c->skip=0;



    /* intra / predictive decision */

    pix = c->src[0][0];

    sum = s->dsp.pix_sum(pix, s->linesize);

    varc = s->dsp.pix_norm1(pix, s->linesize) - (((unsigned)(sum*sum))>>8) + 500;



    pic->mb_mean[s->mb_stride * mb_y + mb_x] = (sum+128)>>8;

    pic->mb_var [s->mb_stride * mb_y + mb_x] = (varc+128)>>8;

    c->mb_var_sum_temp += (varc+128)>>8;



    if(c->avctx->me_threshold){

        vard= check_input_motion(s, mb_x, mb_y, 1);



        if((vard+128)>>8 < c->avctx->me_threshold){

            int p_score= FFMIN(vard, varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*100);

            int i_score= varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*20;

            pic->mc_mb_var[s->mb_stride * mb_y + mb_x] = (vard+128)>>8;

            c->mc_mb_var_sum_temp += (vard+128)>>8;

            c->scene_change_score+= ff_sqrt(p_score) - ff_sqrt(i_score);

            return;

        }

        if((vard+128)>>8 < c->avctx->mb_threshold)

            mb_type= s->mb_type[mb_x + mb_y*s->mb_stride];

    }



    switch(s->me_method) {

    case ME_ZERO:

    default:

        no_motion_search(s, &mx, &my);

        mx-= mb_x*16;

        my-= mb_y*16;

        dmin = 0;

        break;

    case ME_X1:

    case ME_EPZS:

       {

            const int mot_stride = s->b8_stride;

            const int mot_xy = s->block_index[0];



            P_LEFT[0] = s->current_picture.f.motion_val[0][mot_xy - 1][0];

            P_LEFT[1] = s->current_picture.f.motion_val[0][mot_xy - 1][1];



            if(P_LEFT[0]       > (c->xmax<<shift)) P_LEFT[0]       = (c->xmax<<shift);



            if(!s->first_slice_line) {

                P_TOP[0]      = s->current_picture.f.motion_val[0][mot_xy - mot_stride    ][0];

                P_TOP[1]      = s->current_picture.f.motion_val[0][mot_xy - mot_stride    ][1];

                P_TOPRIGHT[0] = s->current_picture.f.motion_val[0][mot_xy - mot_stride + 2][0];

                P_TOPRIGHT[1] = s->current_picture.f.motion_val[0][mot_xy - mot_stride + 2][1];

                if(P_TOP[1]      > (c->ymax<<shift)) P_TOP[1]     = (c->ymax<<shift);

                if(P_TOPRIGHT[0] < (c->xmin<<shift)) P_TOPRIGHT[0]= (c->xmin<<shift);

                if(P_TOPRIGHT[1] > (c->ymax<<shift)) P_TOPRIGHT[1]= (c->ymax<<shift);



                P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

                P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);



                if(s->out_format == FMT_H263){

                    c->pred_x = P_MEDIAN[0];

                    c->pred_y = P_MEDIAN[1];

                }else { /* mpeg1 at least */

                    c->pred_x= P_LEFT[0];

                    c->pred_y= P_LEFT[1];

                }

            }else{

                c->pred_x= P_LEFT[0];

                c->pred_y= P_LEFT[1];

            }



        }

        dmin = ff_epzs_motion_search(s, &mx, &my, P, 0, 0, s->p_mv_table, (1<<16)>>shift, 0, 16);



        break;

    }



    /* At this point (mx,my) are full-pell and the relative displacement */

    ppix = c->ref[0][0] + (my * s->linesize) + mx;



    vard = s->dsp.sse[0](NULL, pix, ppix, s->linesize, 16);



    pic->mc_mb_var[s->mb_stride * mb_y + mb_x] = (vard+128)>>8;

//    pic->mb_cmp_score[s->mb_stride * mb_y + mb_x] = dmin;

    c->mc_mb_var_sum_temp += (vard+128)>>8;



    if(mb_type){

        int p_score= FFMIN(vard, varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*100);

        int i_score= varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*20;

        c->scene_change_score+= ff_sqrt(p_score) - ff_sqrt(i_score);



        if(mb_type == CANDIDATE_MB_TYPE_INTER){

            c->sub_motion_search(s, &mx, &my, dmin, 0, 0, 0, 16);

            set_p_mv_tables(s, mx, my, 1);

        }else{

            mx <<=shift;

            my <<=shift;

        }

        if(mb_type == CANDIDATE_MB_TYPE_INTER4V){

            h263_mv4_search(s, mx, my, shift);



            set_p_mv_tables(s, mx, my, 0);

        }

        if(mb_type == CANDIDATE_MB_TYPE_INTER_I){

            interlaced_search(s, 0, s->p_field_mv_table, s->p_field_select_table, mx, my, 1);

        }

    }else if(c->avctx->mb_decision > FF_MB_DECISION_SIMPLE){

        int p_score= FFMIN(vard, varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*100);

        int i_score= varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*20;

        c->scene_change_score+= ff_sqrt(p_score) - ff_sqrt(i_score);



        if (vard*2 + 200*256 > varc)

            mb_type|= CANDIDATE_MB_TYPE_INTRA;

        if (varc*2 + 200*256 > vard || s->qscale > 24){

//        if (varc*2 + 200*256 + 50*(s->lambda2>>FF_LAMBDA_SHIFT) > vard){

            mb_type|= CANDIDATE_MB_TYPE_INTER;

            c->sub_motion_search(s, &mx, &my, dmin, 0, 0, 0, 16);

            if(s->flags&CODEC_FLAG_MV0)

                if(mx || my)

                    mb_type |= CANDIDATE_MB_TYPE_SKIPPED; //FIXME check difference

        }else{

            mx <<=shift;

            my <<=shift;

        }

        if((s->flags&CODEC_FLAG_4MV)

           && !c->skip && varc>50<<8 && vard>10<<8){

            if(h263_mv4_search(s, mx, my, shift) < INT_MAX)

                mb_type|=CANDIDATE_MB_TYPE_INTER4V;



            set_p_mv_tables(s, mx, my, 0);

        }else

            set_p_mv_tables(s, mx, my, 1);

        if((s->flags&CODEC_FLAG_INTERLACED_ME)

           && !c->skip){ //FIXME varc/d checks

            if(interlaced_search(s, 0, s->p_field_mv_table, s->p_field_select_table, mx, my, 0) < INT_MAX)

                mb_type |= CANDIDATE_MB_TYPE_INTER_I;

        }

    }else{

        int intra_score, i;

        mb_type= CANDIDATE_MB_TYPE_INTER;



        dmin= c->sub_motion_search(s, &mx, &my, dmin, 0, 0, 0, 16);

        if(c->avctx->me_sub_cmp != c->avctx->mb_cmp && !c->skip)

            dmin= ff_get_mb_score(s, mx, my, 0, 0, 0, 16, 1);



        if((s->flags&CODEC_FLAG_4MV)

           && !c->skip && varc>50<<8 && vard>10<<8){

            int dmin4= h263_mv4_search(s, mx, my, shift);

            if(dmin4 < dmin){

                mb_type= CANDIDATE_MB_TYPE_INTER4V;

                dmin=dmin4;

            }

        }

        if((s->flags&CODEC_FLAG_INTERLACED_ME)

           && !c->skip){ //FIXME varc/d checks

            int dmin_i= interlaced_search(s, 0, s->p_field_mv_table, s->p_field_select_table, mx, my, 0);

            if(dmin_i < dmin){

                mb_type = CANDIDATE_MB_TYPE_INTER_I;

                dmin= dmin_i;

            }

        }



//        pic->mb_cmp_score[s->mb_stride * mb_y + mb_x] = dmin;

        set_p_mv_tables(s, mx, my, mb_type!=CANDIDATE_MB_TYPE_INTER4V);



        /* get intra luma score */

        if((c->avctx->mb_cmp&0xFF)==FF_CMP_SSE){

            intra_score= varc - 500;

        }else{

            int mean= (sum+128)>>8;

            mean*= 0x01010101;



            for(i=0; i<16; i++){

                *(uint32_t*)(&c->scratchpad[i*s->linesize+ 0]) = mean;

                *(uint32_t*)(&c->scratchpad[i*s->linesize+ 4]) = mean;

                *(uint32_t*)(&c->scratchpad[i*s->linesize+ 8]) = mean;

                *(uint32_t*)(&c->scratchpad[i*s->linesize+12]) = mean;

            }



            intra_score= s->dsp.mb_cmp[0](s, c->scratchpad, pix, s->linesize, 16);

        }

        intra_score += c->mb_penalty_factor*16;



        if(intra_score < dmin){

            mb_type= CANDIDATE_MB_TYPE_INTRA;

            s->current_picture.f.mb_type[mb_y*s->mb_stride + mb_x] = CANDIDATE_MB_TYPE_INTRA; //FIXME cleanup

        }else

            s->current_picture.f.mb_type[mb_y*s->mb_stride + mb_x] = 0;



        {

            int p_score= FFMIN(vard, varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*100);

            int i_score= varc-500+(s->lambda2>>FF_LAMBDA_SHIFT)*20;

            c->scene_change_score+= ff_sqrt(p_score) - ff_sqrt(i_score);

        }

    }



    s->mb_type[mb_y*s->mb_stride + mb_x]= mb_type;

}
