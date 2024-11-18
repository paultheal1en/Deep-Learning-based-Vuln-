int estimate_motion(MpegEncContext * s,

		    int mb_x, int mb_y,

		    int *mx_ptr, int *my_ptr)

{

    UINT8 *pix, *ppix;

    int sum, varc, vard, mx, my, range, dmin, xx, yy;

    int xmin, ymin, xmax, ymax;

    int rel_xmin, rel_ymin, rel_xmax, rel_ymax;

    int pred_x=0, pred_y=0;

    int P[5][2];

    const int shift= 1+s->quarter_sample;

    

    range = 8 * (1 << (s->f_code - 1));

    /* XXX: temporary kludge to avoid overflow for msmpeg4 */

    if (s->out_format == FMT_H263 && !s->h263_msmpeg4)

	range = range * 2;



    if (s->unrestricted_mv) {

        xmin = -16;

        ymin = -16;

        if (s->h263_plus)

            range *= 2;

        if(s->avctx==NULL || s->avctx->codec->id!=CODEC_ID_MPEG4){

            xmax = s->mb_width*16;

            ymax = s->mb_height*16;

        }else {

            /* XXX: dunno if this is correct but ffmpeg4 decoder wont like it otherwise 

	            (cuz the drawn edge isnt large enough))*/

            xmax = s->width;

            ymax = s->height;

        }

    } else {

        xmin = 0;

        ymin = 0;

        xmax = s->mb_width*16 - 16;

        ymax = s->mb_height*16 - 16;

    }

    switch(s->full_search) {

    case ME_ZERO:

    default:

	no_motion_search(s, &mx, &my);

        dmin = 0;

        break;

    case ME_FULL:

	dmin = full_motion_search(s, &mx, &my, range, xmin, ymin, xmax, ymax);

        break;

    case ME_LOG:

	dmin = log_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax);

        break;

    case ME_PHODS:

	dmin = phods_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax);

        break;

    case ME_X1: // just reserving some space for experiments ...

    case ME_EPZS:

        rel_xmin= xmin - s->mb_x*16;

        rel_xmax= xmax - s->mb_x*16;

        rel_ymin= ymin - s->mb_y*16;

        rel_ymax= ymax - s->mb_y*16;

        if(s->out_format == FMT_H263){

            static const int off[4]= {2, 1, 1, -1};

            const int mot_stride = s->block_wrap[0];

            const int mot_xy = s->block_index[0];

         

            P[0][0] = s->motion_val[mot_xy    ][0];

            P[0][1] = s->motion_val[mot_xy    ][1];

            P[1][0] = s->motion_val[mot_xy - 1][0];

            P[1][1] = s->motion_val[mot_xy - 1][1];

            if(P[1][0] > (rel_xmax<<shift)) P[1][0]= (rel_xmax<<shift);



            /* special case for first line */

            if ((s->mb_y == 0 || s->first_slice_line || s->first_gob_line)) {

                pred_x = P[1][0];

                pred_y = P[1][1];

            } else {

                P[2][0] = s->motion_val[mot_xy - mot_stride             ][0];

                P[2][1] = s->motion_val[mot_xy - mot_stride             ][1];

                P[3][0] = s->motion_val[mot_xy - mot_stride + off[0]    ][0];

                P[3][1] = s->motion_val[mot_xy - mot_stride + off[0]    ][1];

                if(P[2][1] > (rel_ymax<<shift)) P[2][1]= (rel_ymax<<shift);

                if(P[3][0] < (rel_xmin<<shift)) P[3][0]= (rel_xmin<<shift);

                if(P[3][1] > (rel_ymax<<shift)) P[3][1]= (rel_ymax<<shift);

        

                P[4][0]= pred_x = mid_pred(P[1][0], P[2][0], P[3][0]);

                P[4][1]= pred_y = mid_pred(P[1][1], P[2][1], P[3][1]);

            }

        }else {

            const int xy= s->mb_y*s->mb_width + s->mb_x;

            pred_x= s->last_mv[0][0][0];

            pred_y= s->last_mv[0][0][1];



            P[0][0]= s->mv_table[0][xy  ];

            P[0][1]= s->mv_table[1][xy  ];

            if(s->mb_x == 0){

                P[1][0]= 0;

                P[1][1]= 0;

            }else{

                P[1][0]= s->mv_table[0][xy-1];

                P[1][1]= s->mv_table[1][xy-1];

                if(P[1][0] > (rel_xmax<<shift)) P[1][0]= (rel_xmax<<shift);

            }

    

            if (!(s->mb_y == 0 || s->first_slice_line || s->first_gob_line)) {

                P[2][0] = s->mv_table[0][xy - s->mb_width];

                P[2][1] = s->mv_table[1][xy - s->mb_width];

                P[3][0] = s->mv_table[0][xy - s->mb_width+1];

                P[3][1] = s->mv_table[1][xy - s->mb_width+1];

                if(P[2][1] > (rel_ymax<<shift)) P[2][1]= (rel_ymax<<shift);

                if(P[3][0] > (rel_xmax<<shift)) P[3][0]= (rel_xmax<<shift);

                if(P[3][0] < (rel_xmin<<shift)) P[3][0]= (rel_xmin<<shift);

                if(P[3][1] > (rel_ymax<<shift)) P[3][1]= (rel_ymax<<shift);

        

                P[4][0]= mid_pred(P[1][0], P[2][0], P[3][0]);

                P[4][1]= mid_pred(P[1][1], P[2][1], P[3][1]);

            }

        }

	dmin = epzs_motion_search(s, &mx, &my, P, pred_x, pred_y, rel_xmin, rel_ymin, rel_xmax, rel_ymax);

        mx+= s->mb_x*16;

        my+= s->mb_y*16;

        break;

    }



    /* intra / predictive decision */

    xx = mb_x * 16;

    yy = mb_y * 16;



    pix = s->new_picture[0] + (yy * s->linesize) + xx;

    /* At this point (mx,my) are full-pell and the absolute displacement */

    ppix = s->last_picture[0] + (my * s->linesize) + mx;



    sum = pix_sum(pix, s->linesize);

    varc = pix_norm1(pix, s->linesize);

    vard = pix_norm(pix, ppix, s->linesize);



    vard = vard >> 8;

    sum = sum >> 8;

    varc = (varc >> 8) - (sum * sum);

    s->mb_var[s->mb_width * mb_y + mb_x] = varc;

    s->avg_mb_var += varc;

    s->mc_mb_var += vard;

     

#if 0

    printf("varc=%4d avg_var=%4d (sum=%4d) vard=%4d mx=%2d my=%2d\n",

	   varc, s->avg_mb_var, sum, vard, mx - xx, my - yy);

#endif

    if (vard <= 64 || vard < varc) {

        if (s->full_search != ME_ZERO) {

            halfpel_motion_search(s, &mx, &my, dmin, xmin, ymin, xmax, ymax, pred_x, pred_y);

        } else {

            mx -= 16 * s->mb_x;

            my -= 16 * s->mb_y;

        }

	*mx_ptr = mx;

	*my_ptr = my;

	return 0;

    } else {

	*mx_ptr = 0;

	*my_ptr = 0;

	return 1;

    }

}
