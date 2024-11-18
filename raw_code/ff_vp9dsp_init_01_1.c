av_cold void ff_vp9dsp_init(VP9DSPContext *dsp)

{

    vp9dsp_intrapred_init(dsp);

    vp9dsp_itxfm_init(dsp);

    vp9dsp_loopfilter_init(dsp);

    vp9dsp_mc_init(dsp);





    if (ARCH_X86)

        ff_vp9dsp_init_x86(dsp);

}