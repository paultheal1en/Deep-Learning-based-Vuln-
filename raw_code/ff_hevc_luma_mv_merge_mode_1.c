void ff_hevc_luma_mv_merge_mode(HEVCContext *s, int x0, int y0, int nPbW,

                                int nPbH, int log2_cb_size, int part_idx,

                                int merge_idx, MvField *mv)

{

    int singleMCLFlag = 0;

    int nCS = 1 << log2_cb_size;

    LOCAL_ALIGNED(4, MvField, mergecand_list, [MRG_MAX_NUM_CANDS]);

    int nPbW2 = nPbW;

    int nPbH2 = nPbH;

    HEVCLocalContext *lc = &s->HEVClc;



    memset(mergecand_list, 0, MRG_MAX_NUM_CANDS * sizeof(*mergecand_list));



    if (s->pps->log2_parallel_merge_level > 2 && nCS == 8) {

        singleMCLFlag = 1;

        x0            = lc->cu.x;

        y0            = lc->cu.y;

        nPbW          = nCS;

        nPbH          = nCS;

        part_idx      = 0;

    }



    ff_hevc_set_neighbour_available(s, x0, y0, nPbW, nPbH);

    derive_spatial_merge_candidates(s, x0, y0, nPbW, nPbH, log2_cb_size,

                                    singleMCLFlag, part_idx,

                                    merge_idx, mergecand_list);



    if (mergecand_list[merge_idx].pred_flag[0] == 1 &&

        mergecand_list[merge_idx].pred_flag[1] == 1 &&

        (nPbW2 + nPbH2) == 12) {

        mergecand_list[merge_idx].ref_idx[1]   = -1;

        mergecand_list[merge_idx].pred_flag[1] = 0;

    }



    *mv = mergecand_list[merge_idx];

}
