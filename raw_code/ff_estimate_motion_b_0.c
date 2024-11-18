static int ff_estimate_motion_b(MpegEncContext * s,

                       int mb_x, int mb_y, int16_t (*mv_table)[2], Picture *picture, int f_code)

{

    int mx, my, range, dmin;

    int xmin, ymin, xmax, ymax;

    int rel_xmin, rel_ymin, rel_xmax, rel_ymax;

    int pred_x=0, pred_y=0;

    int P[10][2];

    const int shift= 1+s->quarter_sample;

    const int mot_stride = s->mb_width + 2;

    const int mot_xy = (mb_y + 1)*mot_stride + mb_x + 1;

    uint8_t * const ref_picture= picture->data[0];

    uint16_t * const mv_penalty= s->me.mv_penalty[f_code] + MAX_MV;

    int mv_scale;

        

    s->me.penalty_factor    = get_penalty_factor(s, s->avctx->me_cmp);

    s->me.sub_penalty_factor= get_penalty_factor(s, s->avctx->me_sub_cmp);

    s->me.mb_penalty_factor = get_penalty_factor(s, s->avctx->mb_cmp);



    get_limits(s, &range, &xmin, &ymin, &xmax, &ymax, f_code);

    rel_xmin= xmin - mb_x*16;

    rel_xmax= xmax - mb_x*16;

    rel_ymin= ymin - mb_y*16;

    rel_ymax= ymax - mb_y*16;



    switch(s->me_method) {

    case ME_ZERO:

    default:

	no_motion_search(s, &mx, &my);

        dmin = 0;

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

    case ME_FULL:

	dmin = full_motion_search(s, &mx, &my, range, xmin, ymin, xmax, ymax, ref_picture);

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

    case ME_LOG:

	dmin = log_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax, ref_picture);

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

    case ME_PHODS:

	dmin = phods_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax, ref_picture);

        mx-= mb_x*16;

        my-= mb_y*16;

        break;

    case ME_X1:

    case ME_EPZS:

       {

            P_LEFT[0]        = mv_table[mot_xy - 1][0];

            P_LEFT[1]        = mv_table[mot_xy - 1][1];



            if(P_LEFT[0]       > (rel_xmax<<shift)) P_LEFT[0]       = (rel_xmax<<shift);



            /* special case for first line */

            if (mb_y) {

                P_TOP[0] = mv_table[mot_xy - mot_stride             ][0];

                P_TOP[1] = mv_table[mot_xy - mot_stride             ][1];

                P_TOPRIGHT[0] = mv_table[mot_xy - mot_stride + 1         ][0];

                P_TOPRIGHT[1] = mv_table[mot_xy - mot_stride + 1         ][1];

                if(P_TOP[1] > (rel_ymax<<shift)) P_TOP[1]= (rel_ymax<<shift);

                if(P_TOPRIGHT[0] < (rel_xmin<<shift)) P_TOPRIGHT[0]= (rel_xmin<<shift);

                if(P_TOPRIGHT[1] > (rel_ymax<<shift)) P_TOPRIGHT[1]= (rel_ymax<<shift);

        

                P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

                P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);

            }

            pred_x= P_LEFT[0];

            pred_y= P_LEFT[1];

        }

        

        if(mv_table == s->b_forw_mv_table){

            mv_scale= (s->pb_time<<16) / (s->pp_time<<shift);

        }else{

            mv_scale= ((s->pb_time - s->pp_time)<<16) / (s->pp_time<<shift);

        }

        

        dmin = s->me.motion_search[0](s, 0, &mx, &my, P, pred_x, pred_y, rel_xmin, rel_ymin, rel_xmax, rel_ymax, 

                                      picture, s->p_mv_table, mv_scale, mv_penalty);

 

        break;

    }

    

    dmin= s->me.sub_motion_search(s, &mx, &my, dmin, rel_xmin, rel_ymin, rel_xmax, rel_ymax,

				   pred_x, pred_y, picture, 0, 0, mv_penalty);

                                   

    if(s->avctx->me_sub_cmp != s->avctx->mb_cmp && !s->me.skip)

        dmin= s->me.get_mb_score(s, mx, my, pred_x, pred_y, picture, mv_penalty);



//printf("%d %d %d %d//", s->mb_x, s->mb_y, mx, my);

//    s->mb_type[mb_y*s->mb_width + mb_x]= mb_type;

    mv_table[mot_xy][0]= mx;

    mv_table[mot_xy][1]= my;



    return dmin;

}
