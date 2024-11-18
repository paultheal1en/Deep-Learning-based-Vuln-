av_cold void ff_vp3dsp_init_x86(VP3DSPContext *c, int flags)

{

#if HAVE_YASM

    int cpuflags = av_get_cpu_flags();



#if ARCH_X86_32

    if (HAVE_MMX && cpuflags & AV_CPU_FLAG_MMX) {

        c->idct_put  = ff_vp3_idct_put_mmx;

        c->idct_add  = ff_vp3_idct_add_mmx;

        c->idct_perm = FF_PARTTRANS_IDCT_PERM;

    }

#endif



    if (HAVE_MMXEXT && cpuflags & AV_CPU_FLAG_MMXEXT) {

        c->idct_dc_add = ff_vp3_idct_dc_add_mmx2;



        if (!(flags & CODEC_FLAG_BITEXACT)) {

            c->v_loop_filter = ff_vp3_v_loop_filter_mmx2;

            c->h_loop_filter = ff_vp3_h_loop_filter_mmx2;

        }

    }



    if (cpuflags & AV_CPU_FLAG_SSE2) {

        c->idct_put  = ff_vp3_idct_put_sse2;

        c->idct_add  = ff_vp3_idct_add_sse2;

        c->idct_perm = FF_TRANSPOSE_IDCT_PERM;

    }

#endif

}
