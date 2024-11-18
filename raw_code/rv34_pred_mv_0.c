static void rv34_pred_mv(RV34DecContext *r, int block_type, int subblock_no, int dmv_no)

{

    MpegEncContext *s = &r->s;

    int mv_pos = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride;

    int A[2] = {0}, B[2], C[2];

    int i, j;

    int mx, my;

    int avail_index = avail_indexes[subblock_no];

    int c_off = part_sizes_w[block_type];



    mv_pos += (subblock_no & 1) + (subblock_no >> 1)*s->b8_stride;

    if(subblock_no == 3)

        c_off = -1;



    if(r->avail_cache[avail_index - 1]){

        A[0] = s->current_picture_ptr->f.motion_val[0][mv_pos-1][0];

        A[1] = s->current_picture_ptr->f.motion_val[0][mv_pos-1][1];

    }

    if(r->avail_cache[avail_index - 4]){

        B[0] = s->current_picture_ptr->f.motion_val[0][mv_pos-s->b8_stride][0];

        B[1] = s->current_picture_ptr->f.motion_val[0][mv_pos-s->b8_stride][1];

    }else{

        B[0] = A[0];

        B[1] = A[1];

    }

    if(!r->avail_cache[avail_index - 4 + c_off]){

        if(r->avail_cache[avail_index - 4] && (r->avail_cache[avail_index - 1] || r->rv30)){

            C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos-s->b8_stride-1][0];

            C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos-s->b8_stride-1][1];

        }else{

            C[0] = A[0];

            C[1] = A[1];

        }

    }else{

        C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos-s->b8_stride+c_off][0];

        C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos-s->b8_stride+c_off][1];

    }

    mx = mid_pred(A[0], B[0], C[0]);

    my = mid_pred(A[1], B[1], C[1]);

    mx += r->dmv[dmv_no][0];

    my += r->dmv[dmv_no][1];

    for(j = 0; j < part_sizes_h[block_type]; j++){

        for(i = 0; i < part_sizes_w[block_type]; i++){

            s->current_picture_ptr->f.motion_val[0][mv_pos + i + j*s->b8_stride][0] = mx;

            s->current_picture_ptr->f.motion_val[0][mv_pos + i + j*s->b8_stride][1] = my;

        }

    }

}
