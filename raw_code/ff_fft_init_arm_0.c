av_cold void ff_fft_init_arm(FFTContext *s)

{

    if (HAVE_NEON) {

        s->fft_permute  = ff_fft_permute_neon;

        s->fft_calc     = ff_fft_calc_neon;

        s->imdct_calc   = ff_imdct_calc_neon;

        s->imdct_half   = ff_imdct_half_neon;

        s->mdct_calc    = ff_mdct_calc_neon;

        s->permutation  = FF_MDCT_PERM_INTERLEAVE;

    }

}
