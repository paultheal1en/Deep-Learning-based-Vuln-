int ff_estimate_motion_b(MpegEncContext * s,

                       int mb_x, int mb_y, int16_t (*mv_table)[2], uint8_t *ref_picture, int f_code)

{

    int mx, my, range, dmin;

    int xmin, ymin, xmax, ymax;

    int rel_xmin, rel_ymin, rel_xmax, rel_ymax;

    int pred_x=0, pred_y=0;

    int P[6][2];

    const int shift= 1+s->quarter_sample;

    const int mot_stride = s->mb_width + 2;

    const int mot_xy = (mb_y + 1)*mot_stride + mb_x + 1;

    

    get_limits(s, &range, &xmin, &ymin, &xmax, &ymax, f_code);



    switch(s->me_method) {

    case ME_ZERO:

    default:

	no_motion_search(s, &mx, &my);

        dmin = 0;

        break;

    case ME_FULL:

	dmin = full_motion_search(s, &mx, &my, range, xmin, ymin, xmax, ymax, ref_picture);

        break;

    case ME_LOG:

	dmin = log_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax, ref_picture);

        break;

    case ME_PHODS:

	dmin = phods_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax, ref_picture);

        break;

    case ME_X1:

    case ME_EPZS:

       {



            rel_xmin= xmin - mb_x*16;

            rel_xmax= xmax - mb_x*16;

            rel_ymin= ymin - mb_y*16;

            rel_ymax= ymax - mb_y*16;



            P[0][0] = mv_table[mot_xy    ][0];

            P[0][1] = mv_table[mot_xy    ][1];

            P[1][0] = mv_table[mot_xy - 1][0];

            P[1][1] = mv_table[mot_xy - 1][1];

            if(P[1][0] > (rel_xmax<<shift)) P[1][0]= (rel_xmax<<shift);



            /* special case for first line */

            if ((mb_y == 0 || s->first_slice_line || s->first_gob_line)) {

                P[4][0] = P[1][0];

                P[4][1] = P[1][1];

            } else {

                P[2][0] = mv_table[mot_xy - mot_stride             ][0];

                P[2][1] = mv_table[mot_xy - mot_stride             ][1];

                P[3][0] = mv_table[mot_xy - mot_stride + 1         ][0];

                P[3][1] = mv_table[mot_xy - mot_stride + 1         ][1];

                if(P[2][1] > (rel_ymax<<shift)) P[2][1]= (rel_ymax<<shift);

                if(P[3][0] < (rel_xmin<<shift)) P[3][0]= (rel_xmin<<shift);

                if(P[3][1] > (rel_ymax<<shift)) P[3][1]= (rel_ymax<<shift);

        

                P[4][0]= mid_pred(P[1][0], P[2][0], P[3][0]);

                P[4][1]= mid_pred(P[1][1], P[2][1], P[3][1]);

            }

            pred_x= P[1][0];

            pred_y= P[1][1];

        }

        dmin = epzs_motion_search(s, &mx, &my, P, pred_x, pred_y, rel_xmin, rel_ymin, rel_xmax, rel_ymax, ref_picture);

 

        mx+= mb_x*16;

        my+= mb_y*16;

        break;

    }

    

    /* intra / predictive decision */

//    xx = mb_x * 16;

//    yy = mb_y * 16;



//    pix = s->new_picture[0] + (yy * s->linesize) + xx;

    /* At this point (mx,my) are full-pell and the absolute displacement */

//    ppix = ref_picture + (my * s->linesize) + mx;

    

    dmin= halfpel_motion_search(s, &mx, &my, dmin, xmin, ymin, xmax, ymax, pred_x, pred_y, ref_picture);



//    s->mb_type[mb_y*s->mb_width + mb_x]= mb_type;

    mv_table[mot_xy][0]= mx;

    mv_table[mot_xy][1]= my;

    return dmin;

}
