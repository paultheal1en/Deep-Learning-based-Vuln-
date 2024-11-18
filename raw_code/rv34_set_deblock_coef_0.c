static int rv34_set_deblock_coef(RV34DecContext *r)

{

    MpegEncContext *s = &r->s;

    int mvmask = 0, i, j;

    int midx = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride;

    int16_t (*motion_val)[2] = s->current_picture_ptr->motion_val[0][midx];

    if(s->pict_type == FF_I_TYPE)

        return 0;

    for(j = 0; j < 16; j += 8){

        for(i = 0; i < 2; i++){

            if(is_mv_diff_gt_3(motion_val + i, 1))

                mvmask |= 0x11 << (j + i*2);

            if(is_mv_diff_gt_3(motion_val + i, s->b8_stride))

                mvmask |= 0x03 << (j + i*2);

        }

        motion_val += s->b8_stride;

    }

    return mvmask;

}
