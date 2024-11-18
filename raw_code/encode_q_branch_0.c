static int encode_q_branch(SnowContext *s, int level, int x, int y){

    uint8_t p_buffer[1024];

    uint8_t i_buffer[1024];

    uint8_t p_state[sizeof(s->block_state)];

    uint8_t i_state[sizeof(s->block_state)];

    RangeCoder pc, ic;

    uint8_t *pbbak= s->c.bytestream;

    uint8_t *pbbak_start= s->c.bytestream_start;

    int score, score2, iscore, i_len, p_len, block_s, sum, base_bits;

    const int w= s->b_width  << s->block_max_depth;

    const int h= s->b_height << s->block_max_depth;

    const int rem_depth= s->block_max_depth - level;

    const int index= (x + y*w) << rem_depth;

    const int block_w= 1<<(LOG2_MB_SIZE - level);

    int trx= (x+1)<<rem_depth;

    int try= (y+1)<<rem_depth;

    const BlockNode *left  = x ? &s->block[index-1] : &null_block;

    const BlockNode *top   = y ? &s->block[index-w] : &null_block;

    const BlockNode *right = trx<w ? &s->block[index+1] : &null_block;

    const BlockNode *bottom= try<h ? &s->block[index+w] : &null_block;

    const BlockNode *tl    = y && x ? &s->block[index-w-1] : left;

    const BlockNode *tr    = y && trx<w && ((x&1)==0 || level==0) ? &s->block[index-w+(1<<rem_depth)] : tl; //FIXME use lt

    int pl = left->color[0];

    int pcb= left->color[1];

    int pcr= left->color[2];

    int pmx, pmy;

    int mx=0, my=0;

    int l,cr,cb;

    const int stride= s->current_picture->linesize[0];

    const int uvstride= s->current_picture->linesize[1];

    uint8_t *current_data[3]= { s->input_picture->data[0] + (x + y*  stride)*block_w,

                                s->input_picture->data[1] + ((x*block_w)>>s->chroma_h_shift) + ((y*uvstride*block_w)>>s->chroma_v_shift),

                                s->input_picture->data[2] + ((x*block_w)>>s->chroma_h_shift) + ((y*uvstride*block_w)>>s->chroma_v_shift)};

    int P[10][2];

    int16_t last_mv[3][2];

    int qpel= !!(s->avctx->flags & AV_CODEC_FLAG_QPEL); //unused

    const int shift= 1+qpel;

    MotionEstContext *c= &s->m.me;

    int ref_context= av_log2(2*left->ref) + av_log2(2*top->ref);

    int mx_context= av_log2(2*FFABS(left->mx - top->mx));

    int my_context= av_log2(2*FFABS(left->my - top->my));

    int s_context= 2*left->level + 2*top->level + tl->level + tr->level;

    int ref, best_ref, ref_score, ref_mx, ref_my;



    av_assert0(sizeof(s->block_state) >= 256);

    if(s->keyframe){

        set_blocks(s, level, x, y, pl, pcb, pcr, 0, 0, 0, BLOCK_INTRA);

        return 0;

    }



//    clip predictors / edge ?



    P_LEFT[0]= left->mx;

    P_LEFT[1]= left->my;

    P_TOP [0]= top->mx;

    P_TOP [1]= top->my;

    P_TOPRIGHT[0]= tr->mx;

    P_TOPRIGHT[1]= tr->my;



    last_mv[0][0]= s->block[index].mx;

    last_mv[0][1]= s->block[index].my;

    last_mv[1][0]= right->mx;

    last_mv[1][1]= right->my;

    last_mv[2][0]= bottom->mx;

    last_mv[2][1]= bottom->my;



    s->m.mb_stride=2;

    s->m.mb_x=

    s->m.mb_y= 0;

    c->skip= 0;



    av_assert1(c->  stride ==   stride);

    av_assert1(c->uvstride == uvstride);



    c->penalty_factor    = get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_cmp);

    c->sub_penalty_factor= get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_sub_cmp);

    c->mb_penalty_factor = get_penalty_factor(s->lambda, s->lambda2, c->avctx->mb_cmp);

    c->current_mv_penalty= c->mv_penalty[s->m.f_code=1] + MAX_MV;



    c->xmin = - x*block_w - 16+3;

    c->ymin = - y*block_w - 16+3;

    c->xmax = - (x+1)*block_w + (w<<(LOG2_MB_SIZE - s->block_max_depth)) + 16-3;

    c->ymax = - (y+1)*block_w + (h<<(LOG2_MB_SIZE - s->block_max_depth)) + 16-3;



    if(P_LEFT[0]     > (c->xmax<<shift)) P_LEFT[0]    = (c->xmax<<shift);

    if(P_LEFT[1]     > (c->ymax<<shift)) P_LEFT[1]    = (c->ymax<<shift);

    if(P_TOP[0]      > (c->xmax<<shift)) P_TOP[0]     = (c->xmax<<shift);

    if(P_TOP[1]      > (c->ymax<<shift)) P_TOP[1]     = (c->ymax<<shift);

    if(P_TOPRIGHT[0] < (c->xmin<<shift)) P_TOPRIGHT[0]= (c->xmin<<shift);

    if(P_TOPRIGHT[0] > (c->xmax<<shift)) P_TOPRIGHT[0]= (c->xmax<<shift); //due to pmx no clip

    if(P_TOPRIGHT[1] > (c->ymax<<shift)) P_TOPRIGHT[1]= (c->ymax<<shift);



    P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

    P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);



    if (!y) {

        c->pred_x= P_LEFT[0];

        c->pred_y= P_LEFT[1];

    } else {

        c->pred_x = P_MEDIAN[0];

        c->pred_y = P_MEDIAN[1];

    }



    score= INT_MAX;

    best_ref= 0;

    for(ref=0; ref<s->ref_frames; ref++){

        init_ref(c, current_data, s->last_picture[ref]->data, NULL, block_w*x, block_w*y, 0);



        ref_score= ff_epzs_motion_search(&s->m, &ref_mx, &ref_my, P, 0, /*ref_index*/ 0, last_mv,

                                         (1<<16)>>shift, level-LOG2_MB_SIZE+4, block_w);



        av_assert2(ref_mx >= c->xmin);

        av_assert2(ref_mx <= c->xmax);

        av_assert2(ref_my >= c->ymin);

        av_assert2(ref_my <= c->ymax);



        ref_score= c->sub_motion_search(&s->m, &ref_mx, &ref_my, ref_score, 0, 0, level-LOG2_MB_SIZE+4, block_w);

        ref_score= ff_get_mb_score(&s->m, ref_mx, ref_my, 0, 0, level-LOG2_MB_SIZE+4, block_w, 0);

        ref_score+= 2*av_log2(2*ref)*c->penalty_factor;

        if(s->ref_mvs[ref]){

            s->ref_mvs[ref][index][0]= ref_mx;

            s->ref_mvs[ref][index][1]= ref_my;

            s->ref_scores[ref][index]= ref_score;

        }

        if(score > ref_score){

            score= ref_score;

            best_ref= ref;

            mx= ref_mx;

            my= ref_my;

        }

    }

    //FIXME if mb_cmp != SSE then intra cannot be compared currently and mb_penalty vs. lambda2



  //  subpel search

    base_bits= get_rac_count(&s->c) - 8*(s->c.bytestream - s->c.bytestream_start);

    pc= s->c;

    pc.bytestream_start=

    pc.bytestream= p_buffer; //FIXME end/start? and at the other stoo

    memcpy(p_state, s->block_state, sizeof(s->block_state));



    if(level!=s->block_max_depth)

        put_rac(&pc, &p_state[4 + s_context], 1);

    put_rac(&pc, &p_state[1 + left->type + top->type], 0);

    if(s->ref_frames > 1)

        put_symbol(&pc, &p_state[128 + 1024 + 32*ref_context], best_ref, 0);

    pred_mv(s, &pmx, &pmy, best_ref, left, top, tr);

    put_symbol(&pc, &p_state[128 + 32*(mx_context + 16*!!best_ref)], mx - pmx, 1);

    put_symbol(&pc, &p_state[128 + 32*(my_context + 16*!!best_ref)], my - pmy, 1);

    p_len= pc.bytestream - pc.bytestream_start;

    score += (s->lambda2*(get_rac_count(&pc)-base_bits))>>FF_LAMBDA_SHIFT;



    block_s= block_w*block_w;

    sum = pix_sum(current_data[0], stride, block_w, block_w);

    l= (sum + block_s/2)/block_s;

    iscore = pix_norm1(current_data[0], stride, block_w) - 2*l*sum + l*l*block_s;



    if (s->nb_planes > 2) {

        block_s= block_w*block_w>>(s->chroma_h_shift + s->chroma_v_shift);

        sum = pix_sum(current_data[1], uvstride, block_w>>s->chroma_h_shift, block_w>>s->chroma_v_shift);

        cb= (sum + block_s/2)/block_s;

    //    iscore += pix_norm1(&current_mb[1][0], uvstride, block_w>>1) - 2*cb*sum + cb*cb*block_s;

        sum = pix_sum(current_data[2], uvstride, block_w>>s->chroma_h_shift, block_w>>s->chroma_v_shift);

        cr= (sum + block_s/2)/block_s;

    //    iscore += pix_norm1(&current_mb[2][0], uvstride, block_w>>1) - 2*cr*sum + cr*cr*block_s;

    }else

        cb = cr = 0;



    ic= s->c;

    ic.bytestream_start=

    ic.bytestream= i_buffer; //FIXME end/start? and at the other stoo

    memcpy(i_state, s->block_state, sizeof(s->block_state));

    if(level!=s->block_max_depth)

        put_rac(&ic, &i_state[4 + s_context], 1);

    put_rac(&ic, &i_state[1 + left->type + top->type], 1);

    put_symbol(&ic, &i_state[32],  l-pl , 1);

    if (s->nb_planes > 2) {

        put_symbol(&ic, &i_state[64], cb-pcb, 1);

        put_symbol(&ic, &i_state[96], cr-pcr, 1);

    }

    i_len= ic.bytestream - ic.bytestream_start;

    iscore += (s->lambda2*(get_rac_count(&ic)-base_bits))>>FF_LAMBDA_SHIFT;



    av_assert1(iscore < 255*255*256 + s->lambda2*10);

    av_assert1(iscore >= 0);

    av_assert1(l>=0 && l<=255);

    av_assert1(pl>=0 && pl<=255);



    if(level==0){

        int varc= iscore >> 8;

        int vard= score >> 8;

        if (vard <= 64 || vard < varc)

            c->scene_change_score+= ff_sqrt(vard) - ff_sqrt(varc);

        else

            c->scene_change_score+= s->m.qscale;

    }



    if(level!=s->block_max_depth){

        put_rac(&s->c, &s->block_state[4 + s_context], 0);

        score2 = encode_q_branch(s, level+1, 2*x+0, 2*y+0);

        score2+= encode_q_branch(s, level+1, 2*x+1, 2*y+0);

        score2+= encode_q_branch(s, level+1, 2*x+0, 2*y+1);

        score2+= encode_q_branch(s, level+1, 2*x+1, 2*y+1);

        score2+= s->lambda2>>FF_LAMBDA_SHIFT; //FIXME exact split overhead



        if(score2 < score && score2 < iscore)

            return score2;

    }



    if(iscore < score){

        pred_mv(s, &pmx, &pmy, 0, left, top, tr);

        memcpy(pbbak, i_buffer, i_len);

        s->c= ic;

        s->c.bytestream_start= pbbak_start;

        s->c.bytestream= pbbak + i_len;

        set_blocks(s, level, x, y, l, cb, cr, pmx, pmy, 0, BLOCK_INTRA);

        memcpy(s->block_state, i_state, sizeof(s->block_state));

        return iscore;

    }else{

        memcpy(pbbak, p_buffer, p_len);

        s->c= pc;

        s->c.bytestream_start= pbbak_start;

        s->c.bytestream= pbbak + p_len;

        set_blocks(s, level, x, y, pl, pcb, pcr, mx, my, best_ref, 0);

        memcpy(s->block_state, p_state, sizeof(s->block_state));

        return score;

    }

}
