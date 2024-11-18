static av_cold int decode_init(AVCodecContext *avctx)

{

    ASV1Context *const a = avctx->priv_data;

    const int scale      = avctx->codec_id == AV_CODEC_ID_ASV1 ? 1 : 2;

    int i;



    if (avctx->extradata_size < 1) {

        av_log(avctx, AV_LOG_ERROR, "No extradata provided\n");

        return AVERROR_INVALIDDATA;

    }



    ff_asv_common_init(avctx);

    ff_blockdsp_init(&a->bdsp, avctx);

    ff_idctdsp_init(&a->idsp, avctx);

    init_vlcs(a);

    ff_init_scantable(a->idsp.idct_permutation, &a->scantable, ff_asv_scantab);

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;



    a->inv_qscale = avctx->extradata[0];

    if (a->inv_qscale == 0) {

        av_log(avctx, AV_LOG_ERROR, "illegal qscale 0\n");

        if (avctx->codec_id == AV_CODEC_ID_ASV1)

            a->inv_qscale = 6;

        else

            a->inv_qscale = 10;

    }



    for (i = 0; i < 64; i++) {

        int index = ff_asv_scantab[i];



        a->intra_matrix[i] = 64 * scale * ff_mpeg1_default_intra_matrix[index] /

                             a->inv_qscale;

    }



    return 0;

}
