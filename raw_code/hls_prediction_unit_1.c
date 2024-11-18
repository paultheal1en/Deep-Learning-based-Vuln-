static void hls_prediction_unit(HEVCContext *s, int x0, int y0,

                                int nPbW, int nPbH,

                                int log2_cb_size, int partIdx, int idx)

{

#define POS(c_idx, x, y)                                                              \

    &s->frame->data[c_idx][((y) >> s->sps->vshift[c_idx]) * s->frame->linesize[c_idx] + \

                           (((x) >> s->sps->hshift[c_idx]) << s->sps->pixel_shift)]

    HEVCLocalContext *lc = s->HEVClc;

    int merge_idx = 0;

    struct MvField current_mv = {{{ 0 }}};



    int min_pu_width = s->sps->min_pu_width;



    MvField *tab_mvf = s->ref->tab_mvf;

    RefPicList  *refPicList = s->ref->refPicList;

    HEVCFrame *ref0, *ref1;

    uint8_t *dst0 = POS(0, x0, y0);

    uint8_t *dst1 = POS(1, x0, y0);

    uint8_t *dst2 = POS(2, x0, y0);

    int log2_min_cb_size = s->sps->log2_min_cb_size;

    int min_cb_width     = s->sps->min_cb_width;

    int x_cb             = x0 >> log2_min_cb_size;

    int y_cb             = y0 >> log2_min_cb_size;

    int x_pu, y_pu;

    int i, j;



    int skip_flag = SAMPLE_CTB(s->skip_flag, x_cb, y_cb);



    if (!skip_flag)

        lc->pu.merge_flag = ff_hevc_merge_flag_decode(s);



    if (skip_flag || lc->pu.merge_flag) {

        if (s->sh.max_num_merge_cand > 1)

            merge_idx = ff_hevc_merge_idx_decode(s);

        else

            merge_idx = 0;



        ff_hevc_luma_mv_merge_mode(s, x0, y0, nPbW, nPbH, log2_cb_size,

                                   partIdx, merge_idx, &current_mv);

    } else {

        hevc_luma_mv_mpv_mode(s, x0, y0, nPbW, nPbH, log2_cb_size,

                              partIdx, merge_idx, &current_mv);

    }



    x_pu = x0 >> s->sps->log2_min_pu_size;

    y_pu = y0 >> s->sps->log2_min_pu_size;



    for (j = 0; j < nPbH >> s->sps->log2_min_pu_size; j++)

        for (i = 0; i < nPbW >> s->sps->log2_min_pu_size; i++)

            tab_mvf[(y_pu + j) * min_pu_width + x_pu + i] = current_mv;



    if (current_mv.pred_flag & PF_L0) {

        ref0 = refPicList[0].ref[current_mv.ref_idx[0]];

        if (!ref0)

            return;

        hevc_await_progress(s, ref0, &current_mv.mv[0], y0, nPbH);

    }

    if (current_mv.pred_flag & PF_L1) {

        ref1 = refPicList[1].ref[current_mv.ref_idx[1]];

        if (!ref1)

            return;

        hevc_await_progress(s, ref1, &current_mv.mv[1], y0, nPbH);

    }



    if (current_mv.pred_flag == PF_L0) {

        int x0_c = x0 >> s->sps->hshift[1];

        int y0_c = y0 >> s->sps->vshift[1];

        int nPbW_c = nPbW >> s->sps->hshift[1];

        int nPbH_c = nPbH >> s->sps->vshift[1];



        luma_mc_uni(s, dst0, s->frame->linesize[0], ref0->frame,

                    &current_mv.mv[0], x0, y0, nPbW, nPbH,

                    s->sh.luma_weight_l0[current_mv.ref_idx[0]],

                    s->sh.luma_offset_l0[current_mv.ref_idx[0]]);



        chroma_mc_uni(s, dst1, s->frame->linesize[1], ref0->frame->data[1], ref0->frame->linesize[1],

                      0, x0_c, y0_c, nPbW_c, nPbH_c, &current_mv,

                      s->sh.chroma_weight_l0[current_mv.ref_idx[0]][0], s->sh.chroma_offset_l0[current_mv.ref_idx[0]][0]);

        chroma_mc_uni(s, dst2, s->frame->linesize[2], ref0->frame->data[2], ref0->frame->linesize[2],

                      0, x0_c, y0_c, nPbW_c, nPbH_c, &current_mv,

                      s->sh.chroma_weight_l0[current_mv.ref_idx[0]][1], s->sh.chroma_offset_l0[current_mv.ref_idx[0]][1]);

    } else if (current_mv.pred_flag == PF_L1) {

        int x0_c = x0 >> s->sps->hshift[1];

        int y0_c = y0 >> s->sps->vshift[1];

        int nPbW_c = nPbW >> s->sps->hshift[1];

        int nPbH_c = nPbH >> s->sps->vshift[1];



        luma_mc_uni(s, dst0, s->frame->linesize[0], ref1->frame,

                    &current_mv.mv[1], x0, y0, nPbW, nPbH,

                    s->sh.luma_weight_l1[current_mv.ref_idx[1]],

                    s->sh.luma_offset_l1[current_mv.ref_idx[1]]);



        chroma_mc_uni(s, dst1, s->frame->linesize[1], ref1->frame->data[1], ref1->frame->linesize[1],

                      1, x0_c, y0_c, nPbW_c, nPbH_c, &current_mv,

                      s->sh.chroma_weight_l1[current_mv.ref_idx[1]][0], s->sh.chroma_offset_l1[current_mv.ref_idx[1]][0]);



        chroma_mc_uni(s, dst2, s->frame->linesize[2], ref1->frame->data[2], ref1->frame->linesize[2],

                      1, x0_c, y0_c, nPbW_c, nPbH_c, &current_mv,

                      s->sh.chroma_weight_l1[current_mv.ref_idx[1]][1], s->sh.chroma_offset_l1[current_mv.ref_idx[1]][1]);

    } else if (current_mv.pred_flag == PF_BI) {

        int x0_c = x0 >> s->sps->hshift[1];

        int y0_c = y0 >> s->sps->vshift[1];

        int nPbW_c = nPbW >> s->sps->hshift[1];

        int nPbH_c = nPbH >> s->sps->vshift[1];



        luma_mc_bi(s, dst0, s->frame->linesize[0], ref0->frame,

                   &current_mv.mv[0], x0, y0, nPbW, nPbH,

                   ref1->frame, &current_mv.mv[1], &current_mv);



        chroma_mc_bi(s, dst1, s->frame->linesize[1], ref0->frame, ref1->frame,

                     x0_c, y0_c, nPbW_c, nPbH_c, &current_mv, 0);



        chroma_mc_bi(s, dst2, s->frame->linesize[2], ref0->frame, ref1->frame,

                     x0_c, y0_c, nPbW_c, nPbH_c, &current_mv, 1);

    }

}
