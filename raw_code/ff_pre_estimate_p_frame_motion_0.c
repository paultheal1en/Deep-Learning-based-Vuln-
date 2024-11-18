int ff_pre_estimate_p_frame_motion(MpegEncContext * s,

                                    int mb_x, int mb_y)

{

    int mx, my, range, dmin;

    int xmin, ymin, xmax, ymax;

    int rel_xmin, rel_ymin, rel_xmax, rel_ymax;

    int pred_x=0, pred_y=0;

    int P[10][2];

    const int shift= 1+s->quarter_sample;

    uint16_t * const mv_penalty= s->me.mv_penalty[s->f_code] + MAX_MV;

    const int mv_stride= s->mb_width + 2;

    const int xy= mb_x + 1 + (mb_y + 1)*mv_stride;

    

    assert(s->quarter_sample==0 || s->quarter_sample==1);



    s->me.pre_penalty_factor    = get_penalty_factor(s, s->avctx->me_pre_cmp);



    get_limits(s, &range, &xmin, &ymin, &xmax, &ymax, s->f_code);

    rel_xmin= xmin - mb_x*16;

    rel_xmax= xmax - mb_x*16;

    rel_ymin= ymin - mb_y*16;

    rel_ymax= ymax - mb_y*16;

    s->me.skip=0;



    P_LEFT[0]       = s->p_mv_table[xy + 1][0];

    P_LEFT[1]       = s->p_mv_table[xy + 1][1];



    if(P_LEFT[0]       < (rel_xmin<<shift)) P_LEFT[0]       = (rel_xmin<<shift);



    /* special case for first line */

    if (mb_y == s->mb_height-1) {

        pred_x= P_LEFT[0];

        pred_y= P_LEFT[1];

        P_TOP[0]= P_TOPRIGHT[0]= P_MEDIAN[0]=

        P_TOP[1]= P_TOPRIGHT[1]= P_MEDIAN[1]= 0; //FIXME 

    } else {

        P_TOP[0]      = s->p_mv_table[xy + mv_stride    ][0];

        P_TOP[1]      = s->p_mv_table[xy + mv_stride    ][1];

        P_TOPRIGHT[0] = s->p_mv_table[xy + mv_stride - 1][0];

        P_TOPRIGHT[1] = s->p_mv_table[xy + mv_stride - 1][1];

        if(P_TOP[1]      < (rel_ymin<<shift)) P_TOP[1]     = (rel_ymin<<shift);

        if(P_TOPRIGHT[0] > (rel_xmax<<shift)) P_TOPRIGHT[0]= (rel_xmax<<shift);

        if(P_TOPRIGHT[1] < (rel_ymin<<shift)) P_TOPRIGHT[1]= (rel_ymin<<shift);

    

        P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

        P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);



        pred_x = P_MEDIAN[0];

        pred_y = P_MEDIAN[1];

    }

    dmin = s->me.pre_motion_search(s, 0, &mx, &my, P, pred_x, pred_y, rel_xmin, rel_ymin, rel_xmax, rel_ymax, 

                                   &s->last_picture, s->p_mv_table, (1<<16)>>shift, mv_penalty);



    s->p_mv_table[xy][0] = mx<<shift;

    s->p_mv_table[xy][1] = my<<shift;

    

    return dmin;

}
