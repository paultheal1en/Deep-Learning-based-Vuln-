av_cold void ff_hpeldsp_vp3_init_x86(HpelDSPContext *c, int cpu_flags, int flags)

{

    if (EXTERNAL_AMD3DNOW(cpu_flags)) {

        if (flags & AV_CODEC_FLAG_BITEXACT) {

            c->put_no_rnd_pixels_tab[1][1] = ff_put_no_rnd_pixels8_x2_exact_3dnow;

            c->put_no_rnd_pixels_tab[1][2] = ff_put_no_rnd_pixels8_y2_exact_3dnow;

        }

    }



    if (EXTERNAL_MMXEXT(cpu_flags)) {

        if (flags & AV_CODEC_FLAG_BITEXACT) {

            c->put_no_rnd_pixels_tab[1][1] = ff_put_no_rnd_pixels8_x2_exact_mmxext;

            c->put_no_rnd_pixels_tab[1][2] = ff_put_no_rnd_pixels8_y2_exact_mmxext;

        }

    }

}
