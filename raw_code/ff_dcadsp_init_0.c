av_cold void ff_dcadsp_init(DCADSPContext *s)

{

    s->lfe_fir[0] = dca_lfe_fir0_c;

    s->lfe_fir[1] = dca_lfe_fir1_c;

    s->qmf_32_subbands = dca_qmf_32_subbands;

    s->int8x8_fmul_int32 = int8x8_fmul_int32_c;

    if (ARCH_ARM) ff_dcadsp_init_arm(s);

    if (ARCH_X86) ff_dcadsp_init_x86(s);

}
