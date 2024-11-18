static inline int direct_search(MpegEncContext * s,

                                int mb_x, int mb_y)

{

    int P[6][2];

    const int mot_stride = s->mb_width + 2;

    const int mot_xy = (mb_y + 1)*mot_stride + mb_x + 1;

    int dmin, dmin2;

    int motion_fx, motion_fy, motion_bx, motion_by, motion_bx0, motion_by0;

    int motion_dx, motion_dy;

    const int motion_px= s->p_mv_table[mot_xy][0];

    const int motion_py= s->p_mv_table[mot_xy][1];

    const int time_pp= s->pp_time;

    const int time_bp= s->bp_time;

    const int time_pb= time_pp - time_bp;

    int bx, by;

    int mx, my, mx2, my2;

    uint8_t *ref_picture= s->me_scratchpad - (mb_x + 1 + (mb_y + 1)*s->linesize)*16;

    int16_t (*mv_table)[2]= s->b_direct_mv_table;

    uint16_t *mv_penalty= s->mv_penalty[s->f_code] + MAX_MV; // f_code of the prev frame



    /* thanks to iso-mpeg the rounding is different for the zero vector, so we need to handle that ... */

    motion_fx= (motion_px*time_pb)/time_pp;

    motion_fy= (motion_py*time_pb)/time_pp;

    motion_bx0= (-motion_px*time_bp)/time_pp;

    motion_by0= (-motion_py*time_bp)/time_pp;

    motion_dx= motion_dy=0;

    dmin2= check_bidir_mv(s, mb_x, mb_y, 

                          motion_fx, motion_fy,

                          motion_bx0, motion_by0,

                          motion_fx, motion_fy,

                          motion_bx0, motion_by0) - s->qscale;



    motion_bx= motion_fx - motion_px;

    motion_by= motion_fy - motion_py;

    for(by=-1; by<2; by++){

        for(bx=-1; bx<2; bx++){

            uint8_t *dest_y = s->me_scratchpad + (by+1)*s->linesize*16 + (bx+1)*16;

            uint8_t *ptr;

            int dxy;

            int src_x, src_y;

            const int width= s->width;

            const int height= s->height;



            dxy = ((motion_fy & 1) << 1) | (motion_fx & 1);

            src_x = (mb_x + bx) * 16 + (motion_fx >> 1);

            src_y = (mb_y + by) * 16 + (motion_fy >> 1);

            src_x = clip(src_x, -16, width);

            if (src_x == width) dxy &= ~1;

            src_y = clip(src_y, -16, height);

            if (src_y == height) dxy &= ~2;



            ptr = s->last_picture[0] + (src_y * s->linesize) + src_x;

            put_pixels_tab[dxy](dest_y    , ptr    , s->linesize, 16);

            put_pixels_tab[dxy](dest_y + 8, ptr + 8, s->linesize, 16);



            dxy = ((motion_by & 1) << 1) | (motion_bx & 1);

            src_x = (mb_x + bx) * 16 + (motion_bx >> 1);

            src_y = (mb_y + by) * 16 + (motion_by >> 1);

            src_x = clip(src_x, -16, width);

            if (src_x == width) dxy &= ~1;

            src_y = clip(src_y, -16, height);

            if (src_y == height) dxy &= ~2;



            avg_pixels_tab[dxy](dest_y    , ptr    , s->linesize, 16);

            avg_pixels_tab[dxy](dest_y + 8, ptr + 8, s->linesize, 16);

        }

    }



    P[0][0] = mv_table[mot_xy    ][0];

    P[0][1] = mv_table[mot_xy    ][1];

    P[1][0] = mv_table[mot_xy - 1][0];

    P[1][1] = mv_table[mot_xy - 1][1];



    /* special case for first line */

    if ((mb_y == 0 || s->first_slice_line || s->first_gob_line)) {

        P[4][0] = P[1][0];

        P[4][1] = P[1][1];

    } else {

        P[2][0] = mv_table[mot_xy - mot_stride             ][0];

        P[2][1] = mv_table[mot_xy - mot_stride             ][1];

        P[3][0] = mv_table[mot_xy - mot_stride + 1         ][0];

        P[3][1] = mv_table[mot_xy - mot_stride + 1         ][1];

    

        P[4][0]= mid_pred(P[1][0], P[2][0], P[3][0]);

        P[4][1]= mid_pred(P[1][1], P[2][1], P[3][1]);

    }

    dmin = epzs_motion_search(s, &mx, &my, P, 0, 0, -16, -16, 15, 15, ref_picture);

    if(mx==0 && my==0) dmin=99999999; // not representable, due to rounding stuff

    if(dmin2<dmin){ 

        dmin= dmin2;

        mx=0;

        my=0;

    }

#if 1

    mx2= mx= mx*2; 

    my2= my= my*2;

    for(by=-1; by<2; by++){

        if(my2+by < -32) continue;

        for(bx=-1; bx<2; bx++){

            if(bx==0 && by==0) continue;

            if(mx2+bx < -32) continue;

            dmin2= check_bidir_mv(s, mb_x, mb_y, 

                          mx2+bx+motion_fx, my2+by+motion_fy,

                          mx2+bx+motion_bx, my2+by+motion_by,

                          mx2+bx+motion_fx, my2+by+motion_fy,

                          motion_bx, motion_by) - s->qscale;

            

            if(dmin2<dmin){

                dmin=dmin2;

                mx= mx2 + bx;

                my= my2 + by;

            }

        }

    }

#else

    mx*=2; my*=2;

#endif

    if(mx==0 && my==0){

        motion_bx= motion_bx0;

        motion_by= motion_by0;

    }



    s->b_direct_mv_table[mot_xy][0]= mx;

    s->b_direct_mv_table[mot_xy][1]= my;

    s->b_direct_forw_mv_table[mot_xy][0]= motion_fx + mx;

    s->b_direct_forw_mv_table[mot_xy][1]= motion_fy + my;

    s->b_direct_back_mv_table[mot_xy][0]= motion_bx + mx;

    s->b_direct_back_mv_table[mot_xy][1]= motion_by + my;

    return dmin;

}
