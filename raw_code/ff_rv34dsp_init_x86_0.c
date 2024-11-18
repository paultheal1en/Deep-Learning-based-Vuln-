av_cold void ff_rv34dsp_init_x86(RV34DSPContext* c, DSPContext *dsp)

{

#if HAVE_YASM

    int mm_flags = av_get_cpu_flags();



    if (mm_flags & AV_CPU_FLAG_MMX)

        c->rv34_idct_dc_add = ff_rv34_idct_dc_add_mmx;

    if (mm_flags & AV_CPU_FLAG_MMXEXT) {

        c->rv34_inv_transform_dc = ff_rv34_idct_dc_noround_mmx2;

        c->rv34_idct_add         = ff_rv34_idct_add_mmx2;

    }

    if (mm_flags & AV_CPU_FLAG_SSE4)

        c->rv34_idct_dc_add = ff_rv34_idct_dc_add_sse4;

#endif /* HAVE_YASM */

}
