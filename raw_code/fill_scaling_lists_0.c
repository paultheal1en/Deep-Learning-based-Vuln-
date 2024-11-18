static void fill_scaling_lists(const AVCodecContext *avctx, AVDXVAContext *ctx, const H264Context *h, DXVA_Qmatrix_H264 *qm)

{

    unsigned i, j;

    memset(qm, 0, sizeof(*qm));

    if (DXVA_CONTEXT_WORKAROUND(avctx, ctx) & FF_DXVA2_WORKAROUND_SCALING_LIST_ZIGZAG) {

        for (i = 0; i < 6; i++)

            for (j = 0; j < 16; j++)

                qm->bScalingLists4x4[i][j] = h->pps.scaling_matrix4[i][j];



        for (i = 0; i < 64; i++) {

            qm->bScalingLists8x8[0][i] = h->pps.scaling_matrix8[0][i];

            qm->bScalingLists8x8[1][i] = h->pps.scaling_matrix8[3][i];

        }

    } else {

        for (i = 0; i < 6; i++)

            for (j = 0; j < 16; j++)

                qm->bScalingLists4x4[i][j] = h->pps.scaling_matrix4[i][ff_zigzag_scan[j]];



        for (i = 0; i < 64; i++) {

            qm->bScalingLists8x8[0][i] = h->pps.scaling_matrix8[0][ff_zigzag_direct[i]];

            qm->bScalingLists8x8[1][i] = h->pps.scaling_matrix8[3][ff_zigzag_direct[i]];

        }

    }

}
