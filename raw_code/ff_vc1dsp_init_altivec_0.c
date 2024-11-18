void ff_vc1dsp_init_altivec(VC1DSPContext* dsp)

{

    if (!(av_get_cpu_flags() & AV_CPU_FLAG_ALTIVEC))

        return;



    dsp->vc1_inv_trans_8x8 = vc1_inv_trans_8x8_altivec;

    dsp->vc1_inv_trans_8x4 = vc1_inv_trans_8x4_altivec;

    dsp->put_no_rnd_vc1_chroma_pixels_tab[0] = put_no_rnd_vc1_chroma_mc8_altivec;

    dsp->avg_no_rnd_vc1_chroma_pixels_tab[0] = avg_no_rnd_vc1_chroma_mc8_altivec;

}
