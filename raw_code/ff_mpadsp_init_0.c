av_cold void ff_mpadsp_init(MPADSPContext *s)

{

    DCTContext dct;



    ff_dct_init(&dct, 5, DCT_II);

    ff_init_mpadsp_tabs_float();

    ff_init_mpadsp_tabs_fixed();



    s->apply_window_float = ff_mpadsp_apply_window_float;

    s->apply_window_fixed = ff_mpadsp_apply_window_fixed;



    s->dct32_float = dct.dct32;

    s->dct32_fixed = ff_dct32_fixed;



    s->imdct36_blocks_float = ff_imdct36_blocks_float;

    s->imdct36_blocks_fixed = ff_imdct36_blocks_fixed;



    if (ARCH_AARCH64) ff_mpadsp_init_aarch64(s);

    if (ARCH_ARM)     ff_mpadsp_init_arm(s);

    if (ARCH_PPC)     ff_mpadsp_init_ppc(s);

    if (ARCH_X86)     ff_mpadsp_init_x86(s);

    if (HAVE_MIPSFPU)   ff_mpadsp_init_mipsfpu(s);

    if (HAVE_MIPSDSP) ff_mpadsp_init_mipsdsp(s);

}
