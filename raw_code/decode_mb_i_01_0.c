static void decode_mb_i(AVSContext *h) {

    GetBitContext *gb = &h->s.gb;

    int block, pred_mode_uv;

    uint8_t top[18];

    uint8_t left[18];

    uint8_t *d;



    init_mb(h);



    /* get intra prediction modes from stream */

    for(block=0;block<4;block++) {

        int nA,nB,predpred;

        int pos = scan3x3[block];



        nA = h->pred_mode_Y[pos-1];

        nB = h->pred_mode_Y[pos-3];

        if((nA == NOT_AVAIL) || (nB == NOT_AVAIL))

            predpred = 2;

        else

            predpred = FFMIN(nA,nB);

        if(get_bits1(gb))

            h->pred_mode_Y[pos] = predpred;

        else {

            h->pred_mode_Y[pos] = get_bits(gb,2);

            if(h->pred_mode_Y[pos] >= predpred)

                h->pred_mode_Y[pos]++;

        }

    }

    pred_mode_uv = get_ue_golomb(gb);

    if(pred_mode_uv > 6) {

        av_log(h->s.avctx, AV_LOG_ERROR, "illegal intra chroma pred mode\n");

        pred_mode_uv = 0;

    }



    /* save pred modes before they get modified */

    h->pred_mode_Y[3] =  h->pred_mode_Y[5];

    h->pred_mode_Y[6] =  h->pred_mode_Y[8];

    h->top_pred_Y[h->mbx*2+0] = h->pred_mode_Y[7];

    h->top_pred_Y[h->mbx*2+1] = h->pred_mode_Y[8];



    /* modify pred modes according to availability of neighbour samples */

    if(!(h->flags & A_AVAIL)) {

        modify_pred(left_modifier_l, &h->pred_mode_Y[4] );

        modify_pred(left_modifier_l, &h->pred_mode_Y[7] );

        modify_pred(left_modifier_c, &pred_mode_uv );

    }

    if(!(h->flags & B_AVAIL)) {

        modify_pred(top_modifier_l, &h->pred_mode_Y[4] );

        modify_pred(top_modifier_l, &h->pred_mode_Y[5] );

        modify_pred(top_modifier_c, &pred_mode_uv );

    }



    /* get coded block pattern */

    if(h->pic_type == FF_I_TYPE)

        h->cbp = cbp_tab[get_ue_golomb(gb)][0];

    if(h->cbp && !h->qp_fixed)

        h->qp += get_se_golomb(gb); //qp_delta



    /* luma intra prediction interleaved with residual decode/transform/add */

    for(block=0;block<4;block++) {

        d = h->cy + h->luma_scan[block];

        load_intra_pred_luma(h, top, left, block);

        h->intra_pred_l[h->pred_mode_Y[scan3x3[block]]]

            (d, top, left, h->l_stride);

        if(h->cbp & (1<<block))

            decode_residual_block(h,gb,intra_2dvlc,1,h->qp,d,h->l_stride);

    }



    /* chroma intra prediction */

    /* extend borders by one pixel */

    h->left_border_u[9] = h->left_border_u[8];

    h->left_border_v[9] = h->left_border_v[8];

    h->top_border_u[h->mbx*10+9] = h->top_border_u[h->mbx*10+8];

    h->top_border_v[h->mbx*10+9] = h->top_border_v[h->mbx*10+8];

    if(h->mbx && h->mby) {

        h->top_border_u[h->mbx*10] = h->left_border_u[0] = h->topleft_border_u;

        h->top_border_v[h->mbx*10] = h->left_border_v[0] = h->topleft_border_v;

    } else {

        h->left_border_u[0] = h->left_border_u[1];

        h->left_border_v[0] = h->left_border_v[1];

        h->top_border_u[h->mbx*10] = h->top_border_u[h->mbx*10+1];

        h->top_border_v[h->mbx*10] = h->top_border_v[h->mbx*10+1];

    }

    h->intra_pred_c[pred_mode_uv](h->cu, &h->top_border_u[h->mbx*10],

                                  h->left_border_u, h->c_stride);

    h->intra_pred_c[pred_mode_uv](h->cv, &h->top_border_v[h->mbx*10],

                                  h->left_border_v, h->c_stride);



    decode_residual_chroma(h);

    filter_mb(h,I_8X8);



    /* mark motion vectors as intra */

    h->mv[MV_FWD_X0] = intra_mv;

    set_mvs(&h->mv[MV_FWD_X0], BLK_16X16);

    h->mv[MV_BWD_X0] = intra_mv;

    set_mvs(&h->mv[MV_BWD_X0], BLK_16X16);

    if(h->pic_type != FF_B_TYPE)

        *h->col_type = I_8X8;

}
