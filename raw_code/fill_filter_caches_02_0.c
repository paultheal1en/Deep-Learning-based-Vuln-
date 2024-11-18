static int fill_filter_caches(H264Context *h, int mb_type){

    MpegEncContext * const s = &h->s;

    const int mb_xy= h->mb_xy;

    int top_xy, left_xy[2];

    int top_type, left_type[2];



    top_xy     = mb_xy  - (s->mb_stride << MB_FIELD);



    //FIXME deblocking could skip the intra and nnz parts.



    /* Wow, what a mess, why didn't they simplify the interlacing & intra

     * stuff, I can't imagine that these complex rules are worth it. */



    left_xy[1] = left_xy[0] = mb_xy-1;

    if(FRAME_MBAFF){

        const int left_mb_field_flag     = IS_INTERLACED(s->current_picture.mb_type[mb_xy-1]);

        const int curr_mb_field_flag     = IS_INTERLACED(mb_type);

        if(s->mb_y&1){

            if (left_mb_field_flag != curr_mb_field_flag) {

                left_xy[0] -= s->mb_stride;

            }

        }else{

            if(curr_mb_field_flag){

                top_xy      += s->mb_stride & (((s->current_picture.mb_type[top_xy    ]>>7)&1)-1);

            }

            if (left_mb_field_flag != curr_mb_field_flag) {

                left_xy[1] += s->mb_stride;

            }

        }

    }



    h->top_mb_xy = top_xy;

    h->left_mb_xy[0] = left_xy[0];

    h->left_mb_xy[1] = left_xy[1];

    {

        //for sufficiently low qp, filtering wouldn't do anything

        //this is a conservative estimate: could also check beta_offset and more accurate chroma_qp

        int qp_thresh = h->qp_thresh; //FIXME strictly we should store qp_thresh for each mb of a slice

        int qp = s->current_picture.qscale_table[mb_xy];

        if(qp <= qp_thresh

           && (left_xy[0]<0 || ((qp + s->current_picture.qscale_table[left_xy[0]] + 1)>>1) <= qp_thresh)

           && (top_xy   < 0 || ((qp + s->current_picture.qscale_table[top_xy    ] + 1)>>1) <= qp_thresh)){

            if(!FRAME_MBAFF)

                return 1;

            if(   (left_xy[0]< 0            || ((qp + s->current_picture.qscale_table[left_xy[1]             ] + 1)>>1) <= qp_thresh)

               && (top_xy    < s->mb_stride || ((qp + s->current_picture.qscale_table[top_xy    -s->mb_stride] + 1)>>1) <= qp_thresh))

                return 1;

        }

    }



    top_type     = s->current_picture.mb_type[top_xy]    ;

    left_type[0] = s->current_picture.mb_type[left_xy[0]];

    left_type[1] = s->current_picture.mb_type[left_xy[1]];

    if(h->deblocking_filter == 2){

        if(h->slice_table[top_xy     ] != h->slice_num) top_type= 0;

        if(h->slice_table[left_xy[0] ] != h->slice_num) left_type[0]= left_type[1]= 0;

    }else{

        if(h->slice_table[top_xy     ] == 0xFFFF) top_type= 0;

        if(h->slice_table[left_xy[0] ] == 0xFFFF) left_type[0]= left_type[1] =0;

    }

    h->top_type    = top_type    ;

    h->left_type[0]= left_type[0];

    h->left_type[1]= left_type[1];



    if(IS_INTRA(mb_type))

        return 0;



    AV_COPY32(&h->non_zero_count_cache[4+8* 1], &h->non_zero_count[mb_xy][ 0]);

    AV_COPY32(&h->non_zero_count_cache[4+8* 2], &h->non_zero_count[mb_xy][ 4]);

    AV_COPY32(&h->non_zero_count_cache[4+8* 3], &h->non_zero_count[mb_xy][ 8]);

    AV_COPY32(&h->non_zero_count_cache[4+8* 4], &h->non_zero_count[mb_xy][12]);



    h->cbp= h->cbp_table[mb_xy];



    {

        int list;

        for(list=0; list<h->list_count; list++){

            int8_t *ref;

            int y, b_stride;

            int16_t (*mv_dst)[2];

            int16_t (*mv_src)[2];



            if(!USES_LIST(mb_type, list)){

                fill_rectangle(  h->mv_cache[list][scan8[0]], 4, 4, 8, pack16to32(0,0), 4);

                AV_WN32A(&h->ref_cache[list][scan8[ 0]], ((LIST_NOT_USED)&0xFF)*0x01010101u);

                AV_WN32A(&h->ref_cache[list][scan8[ 2]], ((LIST_NOT_USED)&0xFF)*0x01010101u);

                AV_WN32A(&h->ref_cache[list][scan8[ 8]], ((LIST_NOT_USED)&0xFF)*0x01010101u);

                AV_WN32A(&h->ref_cache[list][scan8[10]], ((LIST_NOT_USED)&0xFF)*0x01010101u);

                continue;

            }



            ref = &s->current_picture.ref_index[list][4*mb_xy];

            {

                int (*ref2frm)[64] = h->ref2frm[ h->slice_num&(MAX_SLICES-1) ][0] + (MB_MBAFF ? 20 : 2);

                AV_WN32A(&h->ref_cache[list][scan8[ 0]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);

                AV_WN32A(&h->ref_cache[list][scan8[ 2]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);

                ref += 2;

                AV_WN32A(&h->ref_cache[list][scan8[ 8]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);

                AV_WN32A(&h->ref_cache[list][scan8[10]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);

            }



            b_stride = h->b_stride;

            mv_dst   = &h->mv_cache[list][scan8[0]];

            mv_src   = &s->current_picture.motion_val[list][4*s->mb_x + 4*s->mb_y*b_stride];

            for(y=0; y<4; y++){

                AV_COPY128(mv_dst + 8*y, mv_src + y*b_stride);

            }



        }

    }





/*

0 . T T. T T T T

1 L . .L . . . .

2 L . .L . . . .

3 . T TL . . . .

4 L . .L . . . .

5 L . .. . . . .

*/

//FIXME constraint_intra_pred & partitioning & nnz (let us hope this is just a typo in the spec)

    if(top_type){

        AV_COPY32(&h->non_zero_count_cache[4+8*0], &h->non_zero_count[top_xy][3*4]);

    }



    if(left_type[0]){

        h->non_zero_count_cache[3+8*1]= h->non_zero_count[left_xy[0]][3+0*4];

        h->non_zero_count_cache[3+8*2]= h->non_zero_count[left_xy[0]][3+1*4];

        h->non_zero_count_cache[3+8*3]= h->non_zero_count[left_xy[0]][3+2*4];

        h->non_zero_count_cache[3+8*4]= h->non_zero_count[left_xy[0]][3+3*4];

    }



    // CAVLC 8x8dct requires NNZ values for residual decoding that differ from what the loop filter needs

    if(!CABAC && h->pps.transform_8x8_mode){

        if(IS_8x8DCT(top_type)){

            h->non_zero_count_cache[4+8*0]=

            h->non_zero_count_cache[5+8*0]= (h->cbp_table[top_xy] & 0x4000) >> 12;

            h->non_zero_count_cache[6+8*0]=

            h->non_zero_count_cache[7+8*0]= (h->cbp_table[top_xy] & 0x8000) >> 12;

        }

        if(IS_8x8DCT(left_type[0])){

            h->non_zero_count_cache[3+8*1]=

            h->non_zero_count_cache[3+8*2]= (h->cbp_table[left_xy[0]]&0x2000) >> 12; //FIXME check MBAFF

        }

        if(IS_8x8DCT(left_type[1])){

            h->non_zero_count_cache[3+8*3]=

            h->non_zero_count_cache[3+8*4]= (h->cbp_table[left_xy[1]]&0x8000) >> 12; //FIXME check MBAFF

        }



        if(IS_8x8DCT(mb_type)){

            h->non_zero_count_cache[scan8[0   ]]= h->non_zero_count_cache[scan8[1   ]]=

            h->non_zero_count_cache[scan8[2   ]]= h->non_zero_count_cache[scan8[3   ]]= (h->cbp & 0x1000) >> 12;



            h->non_zero_count_cache[scan8[0+ 4]]= h->non_zero_count_cache[scan8[1+ 4]]=

            h->non_zero_count_cache[scan8[2+ 4]]= h->non_zero_count_cache[scan8[3+ 4]]= (h->cbp & 0x2000) >> 12;



            h->non_zero_count_cache[scan8[0+ 8]]= h->non_zero_count_cache[scan8[1+ 8]]=

            h->non_zero_count_cache[scan8[2+ 8]]= h->non_zero_count_cache[scan8[3+ 8]]= (h->cbp & 0x4000) >> 12;



            h->non_zero_count_cache[scan8[0+12]]= h->non_zero_count_cache[scan8[1+12]]=

            h->non_zero_count_cache[scan8[2+12]]= h->non_zero_count_cache[scan8[3+12]]= (h->cbp & 0x8000) >> 12;

        }

    }



    if(IS_INTER(mb_type) || IS_DIRECT(mb_type)){

        int list;

        for(list=0; list<h->list_count; list++){

            if(USES_LIST(top_type, list)){

                const int b_xy= h->mb2b_xy[top_xy] + 3*h->b_stride;

                const int b8_xy= 4*top_xy + 2;

                int (*ref2frm)[64] = h->ref2frm[ h->slice_table[top_xy]&(MAX_SLICES-1) ][0] + (MB_MBAFF ? 20 : 2);

                AV_COPY128(h->mv_cache[list][scan8[0] + 0 - 1*8], s->current_picture.motion_val[list][b_xy + 0]);

                h->ref_cache[list][scan8[0] + 0 - 1*8]=

                h->ref_cache[list][scan8[0] + 1 - 1*8]= ref2frm[list][s->current_picture.ref_index[list][b8_xy + 0]];

                h->ref_cache[list][scan8[0] + 2 - 1*8]=

                h->ref_cache[list][scan8[0] + 3 - 1*8]= ref2frm[list][s->current_picture.ref_index[list][b8_xy + 1]];

            }else{

                AV_ZERO128(h->mv_cache[list][scan8[0] + 0 - 1*8]);

                AV_WN32A(&h->ref_cache[list][scan8[0] + 0 - 1*8], ((LIST_NOT_USED)&0xFF)*0x01010101u);

            }



            if(!IS_INTERLACED(mb_type^left_type[0])){

                if(USES_LIST(left_type[0], list)){

                    const int b_xy= h->mb2b_xy[left_xy[0]] + 3;

                    const int b8_xy= 4*left_xy[0] + 1;

                    int (*ref2frm)[64] = h->ref2frm[ h->slice_table[left_xy[0]]&(MAX_SLICES-1) ][0] + (MB_MBAFF ? 20 : 2);

                    AV_COPY32(h->mv_cache[list][scan8[0] - 1 + 0 ], s->current_picture.motion_val[list][b_xy + h->b_stride*0]);

                    AV_COPY32(h->mv_cache[list][scan8[0] - 1 + 8 ], s->current_picture.motion_val[list][b_xy + h->b_stride*1]);

                    AV_COPY32(h->mv_cache[list][scan8[0] - 1 +16 ], s->current_picture.motion_val[list][b_xy + h->b_stride*2]);

                    AV_COPY32(h->mv_cache[list][scan8[0] - 1 +24 ], s->current_picture.motion_val[list][b_xy + h->b_stride*3]);

                    h->ref_cache[list][scan8[0] - 1 + 0 ]=

                    h->ref_cache[list][scan8[0] - 1 + 8 ]= ref2frm[list][s->current_picture.ref_index[list][b8_xy + 2*0]];

                    h->ref_cache[list][scan8[0] - 1 +16 ]=

                    h->ref_cache[list][scan8[0] - 1 +24 ]= ref2frm[list][s->current_picture.ref_index[list][b8_xy + 2*1]];

                }else{

                    AV_ZERO32(h->mv_cache [list][scan8[0] - 1 + 0 ]);

                    AV_ZERO32(h->mv_cache [list][scan8[0] - 1 + 8 ]);

                    AV_ZERO32(h->mv_cache [list][scan8[0] - 1 +16 ]);

                    AV_ZERO32(h->mv_cache [list][scan8[0] - 1 +24 ]);

                    h->ref_cache[list][scan8[0] - 1 + 0  ]=

                    h->ref_cache[list][scan8[0] - 1 + 8  ]=

                    h->ref_cache[list][scan8[0] - 1 + 16 ]=

                    h->ref_cache[list][scan8[0] - 1 + 24 ]= LIST_NOT_USED;

                }

            }

        }

    }



    return 0;

}
