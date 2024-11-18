int ff_pred_weight_table(H264Context *h)
{
    int list, i;
    int luma_def, chroma_def;
    h->use_weight             = 0;
    h->use_weight_chroma      = 0;
    h->luma_log2_weight_denom = get_ue_golomb(&h->gb);
    if (h->sps.chroma_format_idc)
        h->chroma_log2_weight_denom = get_ue_golomb(&h->gb);
    luma_def   = 1 << h->luma_log2_weight_denom;
    chroma_def = 1 << h->chroma_log2_weight_denom;
    for (list = 0; list < 2; list++) {
        h->luma_weight_flag[list]   = 0;
        h->chroma_weight_flag[list] = 0;
        for (i = 0; i < h->ref_count[list]; i++) {
            int luma_weight_flag, chroma_weight_flag;
            luma_weight_flag = get_bits1(&h->gb);
            if (luma_weight_flag) {
                h->luma_weight[i][list][0] = get_se_golomb(&h->gb);
                h->luma_weight[i][list][1] = get_se_golomb(&h->gb);
                if (h->luma_weight[i][list][0] != luma_def ||
                    h->luma_weight[i][list][1] != 0) {
                    h->use_weight             = 1;
                    h->luma_weight_flag[list] = 1;
            } else {
                h->luma_weight[i][list][0] = luma_def;
                h->luma_weight[i][list][1] = 0;
            if (h->sps.chroma_format_idc) {
                chroma_weight_flag = get_bits1(&h->gb);
                if (chroma_weight_flag) {
                    int j;
                    for (j = 0; j < 2; j++) {
                        h->chroma_weight[i][list][j][0] = get_se_golomb(&h->gb);
                        h->chroma_weight[i][list][j][1] = get_se_golomb(&h->gb);
                        if (h->chroma_weight[i][list][j][0] != chroma_def ||
                            h->chroma_weight[i][list][j][1] != 0) {
                            h->use_weight_chroma        = 1;
                            h->chroma_weight_flag[list] = 1;
                } else {
                    int j;
                    for (j = 0; j < 2; j++) {
                        h->chroma_weight[i][list][j][0] = chroma_def;
                        h->chroma_weight[i][list][j][1] = 0;
        if (h->slice_type_nos != AV_PICTURE_TYPE_B)
            break;
    h->use_weight = h->use_weight || h->use_weight_chroma;
    return 0;