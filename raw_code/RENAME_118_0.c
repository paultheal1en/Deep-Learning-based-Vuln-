static int RENAME(epzs_motion_search)(MpegEncContext * s,

                             int *mx_ptr, int *my_ptr,

                             int P[10][2], int pred_x, int pred_y, uint8_t *src_data[3], 

                             uint8_t *ref_data[3], int stride, int uvstride, int16_t (*last_mv)[2], 

                             int ref_mv_scale, uint8_t * const mv_penalty)

{

    int best[2]={0, 0};

    int d, dmin; 

    const int shift= 1+s->quarter_sample;

    uint32_t *map= s->me.map;

    int map_generation;

    const int penalty_factor= s->me.penalty_factor;

    const int size=0;

    const int h=16;

    const int ref_mv_stride= s->mb_stride; //pass as arg  FIXME

    const int ref_mv_xy= s->mb_x + s->mb_y*ref_mv_stride; //add to last_mv beforepassing FIXME

    me_cmp_func cmp, chroma_cmp;

    LOAD_COMMON

    

    cmp= s->dsp.me_cmp[size];

    chroma_cmp= s->dsp.me_cmp[size+1];

    

    map_generation= update_map_generation(s);



    CMP(dmin, 0, 0, size);

    map[0]= map_generation;

    score_map[0]= dmin;



    /* first line */

    if (s->first_slice_line) {

        CHECK_MV(P_LEFT[0]>>shift, P_LEFT[1]>>shift)

        CHECK_CLIPED_MV((last_mv[ref_mv_xy][0]*ref_mv_scale + (1<<15))>>16, 

                        (last_mv[ref_mv_xy][1]*ref_mv_scale + (1<<15))>>16)

    }else{

        if(dmin<256 && ( P_LEFT[0]    |P_LEFT[1]

                        |P_TOP[0]     |P_TOP[1]

                        |P_TOPRIGHT[0]|P_TOPRIGHT[1])==0){

            *mx_ptr= 0;

            *my_ptr= 0;

            s->me.skip=1;

            return dmin;

        }

        CHECK_MV(P_MEDIAN[0]>>shift, P_MEDIAN[1]>>shift)

        if(dmin>256*2){

            CHECK_CLIPED_MV((last_mv[ref_mv_xy][0]*ref_mv_scale + (1<<15))>>16, 

                            (last_mv[ref_mv_xy][1]*ref_mv_scale + (1<<15))>>16)

            CHECK_MV(P_LEFT[0]    >>shift, P_LEFT[1]    >>shift)

            CHECK_MV(P_TOP[0]     >>shift, P_TOP[1]     >>shift)

            CHECK_MV(P_TOPRIGHT[0]>>shift, P_TOPRIGHT[1]>>shift)

        }

    }

    if(dmin>256*4){

        if(s->me.pre_pass){

            CHECK_CLIPED_MV((last_mv[ref_mv_xy-1][0]*ref_mv_scale + (1<<15))>>16, 

                            (last_mv[ref_mv_xy-1][1]*ref_mv_scale + (1<<15))>>16)

            if(!s->first_slice_line)

                CHECK_CLIPED_MV((last_mv[ref_mv_xy-ref_mv_stride][0]*ref_mv_scale + (1<<15))>>16, 

                                (last_mv[ref_mv_xy-ref_mv_stride][1]*ref_mv_scale + (1<<15))>>16)

        }else{

            CHECK_CLIPED_MV((last_mv[ref_mv_xy+1][0]*ref_mv_scale + (1<<15))>>16, 

                            (last_mv[ref_mv_xy+1][1]*ref_mv_scale + (1<<15))>>16)

            if(s->end_mb_y == s->mb_height || s->mb_y+1<s->end_mb_y)  //FIXME replace at least with last_slice_line

                CHECK_CLIPED_MV((last_mv[ref_mv_xy+ref_mv_stride][0]*ref_mv_scale + (1<<15))>>16, 

                                (last_mv[ref_mv_xy+ref_mv_stride][1]*ref_mv_scale + (1<<15))>>16)

        }

    }



    if(s->avctx->last_predictor_count){

        const int count= s->avctx->last_predictor_count;

        const int xstart= FFMAX(0, s->mb_x - count);

        const int ystart= FFMAX(0, s->mb_y - count);

        const int xend= FFMIN(s->mb_width , s->mb_x + count + 1);

        const int yend= FFMIN(s->mb_height, s->mb_y + count + 1);

        int mb_y;



        for(mb_y=ystart; mb_y<yend; mb_y++){

            int mb_x;

            for(mb_x=xstart; mb_x<xend; mb_x++){

                const int xy= mb_x + 1 + (mb_y + 1)*ref_mv_stride;

                int mx= (last_mv[xy][0]*ref_mv_scale + (1<<15))>>16;

                int my= (last_mv[xy][1]*ref_mv_scale + (1<<15))>>16;



                if(mx>xmax || mx<xmin || my>ymax || my<ymin) continue;

                CHECK_MV(mx,my)

            }

        }

    }



//check(best[0],best[1],0, b0)

    if(s->me.dia_size==-1)

        dmin= RENAME(funny_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);

    else if(s->me.dia_size<-1)

        dmin= RENAME(sab_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);

    else if(s->me.dia_size<2)

        dmin= RENAME(small_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);

    else

        dmin= RENAME(var_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);



//check(best[0],best[1],0, b1)

    *mx_ptr= best[0];

    *my_ptr= best[1];    



//    printf("%d %d %d \n", best[0], best[1], dmin);

    return dmin;

}
