static inline void write_back_motion(H264Context *h, int mb_type){

    MpegEncContext * const s = &h->s;

    const int b_xy = 4*s->mb_x + 4*s->mb_y*h->b_stride;

    const int b8_xy= 2*s->mb_x + 2*s->mb_y*h->b8_stride;

    int list;



    if(!USES_LIST(mb_type, 0))

        fill_rectangle(&s->current_picture.ref_index[0][b8_xy], 2, 2, h->b8_stride, (uint8_t)LIST_NOT_USED, 1);



    for(list=0; list<2; list++){

        int y;

        if(!USES_LIST(mb_type, list))

            continue;



        for(y=0; y<4; y++){

            *(uint64_t*)s->current_picture.motion_val[list][b_xy + 0 + y*h->b_stride]= *(uint64_t*)h->mv_cache[list][scan8[0]+0 + 8*y];

            *(uint64_t*)s->current_picture.motion_val[list][b_xy + 2 + y*h->b_stride]= *(uint64_t*)h->mv_cache[list][scan8[0]+2 + 8*y];

        }

        if( h->pps.cabac ) {

            if(IS_SKIP(mb_type))

                fill_rectangle(h->mvd_table[list][b_xy], 4, 4, h->b_stride, 0, 4);

            else

            for(y=0; y<4; y++){

                *(uint64_t*)h->mvd_table[list][b_xy + 0 + y*h->b_stride]= *(uint64_t*)h->mvd_cache[list][scan8[0]+0 + 8*y];

                *(uint64_t*)h->mvd_table[list][b_xy + 2 + y*h->b_stride]= *(uint64_t*)h->mvd_cache[list][scan8[0]+2 + 8*y];

            }

        }



        {

            int8_t *ref_index = &s->current_picture.ref_index[list][b8_xy];

            ref_index[0+0*h->b8_stride]= h->ref_cache[list][scan8[0]];

            ref_index[1+0*h->b8_stride]= h->ref_cache[list][scan8[4]];

            ref_index[0+1*h->b8_stride]= h->ref_cache[list][scan8[8]];

            ref_index[1+1*h->b8_stride]= h->ref_cache[list][scan8[12]];

        }

    }



    if(h->slice_type == B_TYPE && h->pps.cabac){

        if(IS_8X8(mb_type)){

            uint8_t *direct_table = &h->direct_table[b8_xy];

            direct_table[1+0*h->b8_stride] = IS_DIRECT(h->sub_mb_type[1]) ? 1 : 0;

            direct_table[0+1*h->b8_stride] = IS_DIRECT(h->sub_mb_type[2]) ? 1 : 0;

            direct_table[1+1*h->b8_stride] = IS_DIRECT(h->sub_mb_type[3]) ? 1 : 0;

        }

    }

}
