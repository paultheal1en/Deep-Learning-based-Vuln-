int ff_h264_alloc_tables(H264Context *h){

    MpegEncContext * const s = &h->s;

    const int big_mb_num= s->mb_stride * (s->mb_height+1);

    const int row_mb_num= 2*s->mb_stride*s->avctx->thread_count;

    int x,y;



    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->intra4x4_pred_mode, row_mb_num * 8  * sizeof(uint8_t), fail)



    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->non_zero_count    , big_mb_num * 48 * sizeof(uint8_t), fail)

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->slice_table_base  , (big_mb_num+s->mb_stride) * sizeof(*h->slice_table_base), fail)

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->cbp_table, big_mb_num * sizeof(uint16_t), fail)



    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->chroma_pred_mode_table, big_mb_num * sizeof(uint8_t), fail)

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->mvd_table[0], 16*row_mb_num * sizeof(uint8_t), fail);

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->mvd_table[1], 16*row_mb_num * sizeof(uint8_t), fail);

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->direct_table, 4*big_mb_num * sizeof(uint8_t) , fail);

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->list_counts, big_mb_num * sizeof(uint8_t), fail)



    memset(h->slice_table_base, -1, (big_mb_num+s->mb_stride)  * sizeof(*h->slice_table_base));

    h->slice_table= h->slice_table_base + s->mb_stride*2 + 1;



    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->mb2b_xy  , big_mb_num * sizeof(uint32_t), fail);

    FF_ALLOCZ_OR_GOTO(h->s.avctx, h->mb2br_xy , big_mb_num * sizeof(uint32_t), fail);

    for(y=0; y<s->mb_height; y++){

        for(x=0; x<s->mb_width; x++){

            const int mb_xy= x + y*s->mb_stride;

            const int b_xy = 4*x + 4*y*h->b_stride;



            h->mb2b_xy [mb_xy]= b_xy;

            h->mb2br_xy[mb_xy]= 8*(FMO ? mb_xy : (mb_xy % (2*s->mb_stride)));

        }

    }



    s->obmc_scratchpad = NULL;



    if(!h->dequant4_coeff[0])

        init_dequant_tables(h);



    return 0;

fail:

    free_tables(h, 1);

    return -1;

}
