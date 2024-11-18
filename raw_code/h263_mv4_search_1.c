static inline int h263_mv4_search(MpegEncContext *s, int mx, int my, int shift)

{

    MotionEstContext * const c= &s->me;

    const int size= 1;

    const int h=8;

    int block;

    int P[10][2];

    int dmin_sum=0, mx4_sum=0, my4_sum=0;

    int same=1;

    const int stride= c->stride;

    uint8_t *mv_penalty= c->current_mv_penalty;



    init_mv4_ref(c);



    for(block=0; block<4; block++){

        int mx4, my4;

        int pred_x4, pred_y4;

        int dmin4;

        static const int off[4]= {2, 1, 1, -1};

        const int mot_stride = s->b8_stride;

        const int mot_xy = s->block_index[block];



        P_LEFT[0] = s->current_picture.motion_val[0][mot_xy - 1][0];

        P_LEFT[1] = s->current_picture.motion_val[0][mot_xy - 1][1];



        if(P_LEFT[0]       > (c->xmax<<shift)) P_LEFT[0]       = (c->xmax<<shift);



        /* special case for first line */

        if (s->first_slice_line && block<2) {

            c->pred_x= pred_x4= P_LEFT[0];

            c->pred_y= pred_y4= P_LEFT[1];

        } else {

            P_TOP[0]      = s->current_picture.motion_val[0][mot_xy - mot_stride             ][0];

            P_TOP[1]      = s->current_picture.motion_val[0][mot_xy - mot_stride             ][1];

            P_TOPRIGHT[0] = s->current_picture.motion_val[0][mot_xy - mot_stride + off[block]][0];

            P_TOPRIGHT[1] = s->current_picture.motion_val[0][mot_xy - mot_stride + off[block]][1];

            if(P_TOP[1]      > (c->ymax<<shift)) P_TOP[1]     = (c->ymax<<shift);

            if(P_TOPRIGHT[0] < (c->xmin<<shift)) P_TOPRIGHT[0]= (c->xmin<<shift);

            if(P_TOPRIGHT[0] > (c->xmax<<shift)) P_TOPRIGHT[0]= (c->xmax<<shift);

            if(P_TOPRIGHT[1] > (c->ymax<<shift)) P_TOPRIGHT[1]= (c->ymax<<shift);



            P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

            P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);



            c->pred_x= pred_x4 = P_MEDIAN[0];

            c->pred_y= pred_y4 = P_MEDIAN[1];

        }

        P_MV1[0]= mx;

        P_MV1[1]= my;



        dmin4 = epzs_motion_search4(s, &mx4, &my4, P, block, block, s->p_mv_table, (1<<16)>>shift);



        dmin4= c->sub_motion_search(s, &mx4, &my4, dmin4, block, block, size, h);



        if(s->dsp.me_sub_cmp[0] != s->dsp.mb_cmp[0]){

            int dxy;

            const int offset= ((block&1) + (block>>1)*stride)*8;

            uint8_t *dest_y = c->scratchpad + offset;

            if(s->quarter_sample){

                uint8_t *ref= c->ref[block][0] + (mx4>>2) + (my4>>2)*stride;

                dxy = ((my4 & 3) << 2) | (mx4 & 3);



                if(s->no_rounding)

                    s->dsp.put_no_rnd_qpel_pixels_tab[1][dxy](dest_y   , ref    , stride);

                else

                    s->dsp.put_qpel_pixels_tab       [1][dxy](dest_y   , ref    , stride);

            }else{

                uint8_t *ref= c->ref[block][0] + (mx4>>1) + (my4>>1)*stride;

                dxy = ((my4 & 1) << 1) | (mx4 & 1);



                if(s->no_rounding)

                    s->hdsp.put_no_rnd_pixels_tab[1][dxy](dest_y    , ref    , stride, h);

                else

                    s->hdsp.put_pixels_tab       [1][dxy](dest_y    , ref    , stride, h);

            }

            dmin_sum+= (mv_penalty[mx4-pred_x4] + mv_penalty[my4-pred_y4])*c->mb_penalty_factor;

        }else

            dmin_sum+= dmin4;



        if(s->quarter_sample){

            mx4_sum+= mx4/2;

            my4_sum+= my4/2;

        }else{

            mx4_sum+= mx4;

            my4_sum+= my4;

        }



        s->current_picture.motion_val[0][s->block_index[block]][0] = mx4;

        s->current_picture.motion_val[0][s->block_index[block]][1] = my4;



        if(mx4 != mx || my4 != my) same=0;

    }



    if(same)

        return INT_MAX;



    if(s->dsp.me_sub_cmp[0] != s->dsp.mb_cmp[0]){

        dmin_sum += s->dsp.mb_cmp[0](s, s->new_picture.f.data[0] + s->mb_x*16 + s->mb_y*16*stride, c->scratchpad, stride, 16);

    }



    if(c->avctx->mb_cmp&FF_CMP_CHROMA){

        int dxy;

        int mx, my;

        int offset;



        mx= ff_h263_round_chroma(mx4_sum);

        my= ff_h263_round_chroma(my4_sum);

        dxy = ((my & 1) << 1) | (mx & 1);



        offset= (s->mb_x*8 + (mx>>1)) + (s->mb_y*8 + (my>>1))*s->uvlinesize;



        if(s->no_rounding){

            s->hdsp.put_no_rnd_pixels_tab[1][dxy](c->scratchpad    , s->last_picture.f.data[1] + offset, s->uvlinesize, 8);

            s->hdsp.put_no_rnd_pixels_tab[1][dxy](c->scratchpad + 8, s->last_picture.f.data[2] + offset, s->uvlinesize, 8);

        }else{

            s->hdsp.put_pixels_tab       [1][dxy](c->scratchpad    , s->last_picture.f.data[1] + offset, s->uvlinesize, 8);

            s->hdsp.put_pixels_tab       [1][dxy](c->scratchpad + 8, s->last_picture.f.data[2] + offset, s->uvlinesize, 8);

        }



        dmin_sum += s->dsp.mb_cmp[1](s, s->new_picture.f.data[1] + s->mb_x*8 + s->mb_y*8*s->uvlinesize, c->scratchpad  , s->uvlinesize, 8);

        dmin_sum += s->dsp.mb_cmp[1](s, s->new_picture.f.data[2] + s->mb_x*8 + s->mb_y*8*s->uvlinesize, c->scratchpad+8, s->uvlinesize, 8);

    }



    c->pred_x= mx;

    c->pred_y= my;



    switch(c->avctx->mb_cmp&0xFF){

    /*case FF_CMP_SSE:

        return dmin_sum+ 32*s->qscale*s->qscale;*/

    case FF_CMP_RD:

        return dmin_sum;

    default:

        return dmin_sum+ 11*c->mb_penalty_factor;

    }

}
