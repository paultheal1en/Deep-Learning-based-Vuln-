void MPV_common_init_altivec(MpegEncContext *s)
{
    if (s->avctx->lowres==0)
    {
        if ((s->avctx->idct_algo == FF_IDCT_AUTO) ||
                (s->avctx->idct_algo == FF_IDCT_ALTIVEC))
        {
            s->dsp.idct_put = idct_put_altivec;
            s->dsp.idct_add = idct_add_altivec;
            s->dsp.idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;
        }
    }
    // Test to make sure that the dct required alignments are met.
    if ((((long)(s->q_intra_matrix) & 0x0f) != 0) ||
        (((long)(s->q_inter_matrix) & 0x0f) != 0))
    {
        av_log(s->avctx, AV_LOG_INFO, "Internal Error: q-matrix blocks must be 16-byte aligned "
                "to use AltiVec DCT. Reverting to non-AltiVec version.\n");
        return;
    }
    if (((long)(s->intra_scantable.inverse) & 0x0f) != 0)
    {
        av_log(s->avctx, AV_LOG_INFO, "Internal Error: scan table blocks must be 16-byte aligned "
                "to use AltiVec DCT. Reverting to non-AltiVec version.\n");
        return;
    }
    if ((s->avctx->dct_algo == FF_DCT_AUTO) ||
            (s->avctx->dct_algo == FF_DCT_ALTIVEC))
    {
#if 0 /* seems to cause trouble under some circumstances */
        s->dct_quantize = dct_quantize_altivec;
#endif
        s->dct_unquantize_h263_intra = dct_unquantize_h263_altivec;
        s->dct_unquantize_h263_inter = dct_unquantize_h263_altivec;
    }
}