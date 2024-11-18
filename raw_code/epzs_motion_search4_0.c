static int epzs_motion_search4(MpegEncContext * s, int block,

                             int *mx_ptr, int *my_ptr,

                             int P[6][2], int pred_x, int pred_y,

                             int xmin, int ymin, int xmax, int ymax, uint8_t *ref_picture)

{

    int best[2]={0, 0};

    int d, dmin; 

    UINT8 *new_pic, *old_pic;

    const int pic_stride= s->linesize;

    const int pic_xy= ((s->mb_y*2 + (block>>1))*pic_stride + s->mb_x*2 + (block&1))*8;

    UINT16 *mv_penalty= s->mv_penalty[s->f_code] + MAX_MV; // f_code of the prev frame

    int quant= s->qscale; // qscale of the prev frame

    const int shift= 1+s->quarter_sample;



    new_pic = s->new_picture[0] + pic_xy;

    old_pic = ref_picture + pic_xy;

   

    dmin = pix_abs8x8(new_pic, old_pic, pic_stride);



    /* first line */

    if ((s->mb_y == 0 || s->first_slice_line || s->first_gob_line) && block<2) {

        CHECK_MV4(P[1][0]>>shift, P[1][1]>>shift)

    }else{

        CHECK_MV4(P[4][0]>>shift, P[4][1]>>shift)

        if(dmin<Z_THRESHOLD){

            *mx_ptr= P[4][0]>>shift;

            *my_ptr= P[4][1]>>shift;

//printf("M\n");

            return dmin;

        }

        CHECK_MV4(P[1][0]>>shift, P[1][1]>>shift)

        CHECK_MV4(P[2][0]>>shift, P[2][1]>>shift)

        CHECK_MV4(P[3][0]>>shift, P[3][1]>>shift)

    }

    CHECK_MV4(P[0][0]>>shift, P[0][1]>>shift)

    CHECK_MV4(P[5][0]>>shift, P[5][1]>>shift)



//check(best[0],best[1],0, b0)

    dmin= small_diamond_search4MV(s, best, dmin, new_pic, old_pic, pic_stride, 

                                   pred_x, pred_y, mv_penalty, quant, xmin, ymin, xmax, ymax, shift);

//check(best[0],best[1],0, b1)

    *mx_ptr= best[0];

    *my_ptr= best[1];    



//    printf("%d %d %d \n", best[0], best[1], dmin);

    return dmin;

}
