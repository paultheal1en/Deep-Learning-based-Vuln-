static void clone_tables(H264Context *dst, H264Context *src){

    dst->intra4x4_pred_mode       = src->intra4x4_pred_mode;

    dst->non_zero_count           = src->non_zero_count;

    dst->slice_table              = src->slice_table;

    dst->cbp_table                = src->cbp_table;

    dst->mb2b_xy                  = src->mb2b_xy;

    dst->mb2b8_xy                 = src->mb2b8_xy;

    dst->chroma_pred_mode_table   = src->chroma_pred_mode_table;

    dst->mvd_table[0]             = src->mvd_table[0];

    dst->mvd_table[1]             = src->mvd_table[1];

    dst->direct_table             = src->direct_table;



    if(!dst->dequant4_coeff[0])

        init_dequant_tables(dst);

    dst->s.obmc_scratchpad = NULL;

    ff_h264_pred_init(&dst->hpc, src->s.codec_id);

    dst->dequant_coeff_pps= -1;

}
