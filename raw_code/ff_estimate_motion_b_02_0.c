static int ff_estimate_motion_b(MpegEncContext * s,

                       int mb_x, int mb_y, int16_t (*mv_table)[2], int ref_index, int f_code)

{

    MotionEstContext * const c= &s->me;

    int mx, my, dmin;

    int P[10][2];

    const int shift= 1+s->quarter_sample;

    const int mot_stride = s->mb_stride;

    const int mot_xy = mb_y*mot_stride + mb_x;

    uint8_t * const mv_penalty= c->mv_penalty[f_code] + MAX_MV;

    int mv_scale;

        

    c->penalty_factor    = get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_cmp);

    c->sub_penalty_factor= get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_sub_cmp);

    c->mb_penalty_factor = get_penalty_factor(s->lambda, s->lambda2, c->avctx->mb_cmp);

    c->current_mv_penalty= mv_penalty;



    get_limits(s, 16*mb_x, 16*mb_y);



    switch(s->me_method) {

    case ME_ZERO:

    default:

	no_motion_search(s, &mx, &my);

        dmin = 0;

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

#if 0

    case ME_FULL:

	dmin = full_motion_search(s, &mx, &my, range, ref_picture);

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

    case ME_LOG:

	dmin = log_motion_search(s, &mx, &my, range / 2, ref_picture);

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

    case ME_PHODS:

	dmin = phods_motion_search(s, &mx, &my, range / 2, ref_picture);

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

#endif

    case ME_X1:

    case ME_EPZS:

       {

            P_LEFT[0]        = mv_table[mot_xy - 1][0];

            P_LEFT[1]        = mv_table[mot_xy - 1][1];



            if(P_LEFT[0]       > (c->xmax<<shift)) P_LEFT[0]       = (c->xmax<<shift);



            /* special case for first line */

            if (!s->first_slice_line) {

                P_TOP[0] = mv_table[mot_xy - mot_stride             ][0];

                P_TOP[1] = mv_table[mot_xy - mot_stride             ][1];

                P_TOPRIGHT[0] = mv_table[mot_xy - mot_stride + 1         ][0];

                P_TOPRIGHT[1] = mv_table[mot_xy - mot_stride + 1         ][1];

                if(P_TOP[1] > (c->ymax<<shift)) P_TOP[1]= (c->ymax<<shift);

                if(P_TOPRIGHT[0] < (c->xmin<<shift)) P_TOPRIGHT[0]= (c->xmin<<shift);

                if(P_TOPRIGHT[1] > (c->ymax<<shift)) P_TOPRIGHT[1]= (c->ymax<<shift);

        

                P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

                P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);

            }

            c->pred_x= P_LEFT[0];

            c->pred_y= P_LEFT[1];

        }

        

        if(mv_table == s->b_forw_mv_table){

            mv_scale= (s->pb_time<<16) / (s->pp_time<<shift);

        }else{

            mv_scale= ((s->pb_time - s->pp_time)<<16) / (s->pp_time<<shift);

        }

        

        dmin = ff_epzs_motion_search(s, &mx, &my, P, 0, ref_index, s->p_mv_table, mv_scale, 0, 16);

 

        break;

    }

    

    dmin= c->sub_motion_search(s, &mx, &my, dmin, 0, ref_index, 0, 16);

                                   

    if(c->avctx->me_sub_cmp != c->avctx->mb_cmp && !c->skip)

        dmin= get_mb_score(s, mx, my, 0, ref_index);



//printf("%d %d %d %d//", s->mb_x, s->mb_y, mx, my);

//    s->mb_type[mb_y*s->mb_width + mb_x]= mb_type;

    mv_table[mot_xy][0]= mx;

    mv_table[mot_xy][1]= my;



    return dmin;

}
