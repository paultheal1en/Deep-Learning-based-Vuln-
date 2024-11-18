av_cold void ff_fft_init_arm(FFTContext *s)

{

    int cpu_flags = av_get_cpu_flags();



    if (have_vfp(cpu_flags) && !have_vfpv3(cpu_flags)) {

        s->fft_calc     = ff_fft_calc_vfp;

#if CONFIG_MDCT

        s->imdct_half   = ff_imdct_half_vfp;

#endif

    }



    if (have_neon(cpu_flags)) {

        s->fft_permute  = ff_fft_permute_neon;

        s->fft_calc     = ff_fft_calc_neon;

#if CONFIG_MDCT

        s->imdct_calc   = ff_imdct_calc_neon;

        s->imdct_half   = ff_imdct_half_neon;

        s->mdct_calc    = ff_mdct_calc_neon;

        s->mdct_permutation = FF_MDCT_PERM_INTERLEAVE;

#endif

    }

}
