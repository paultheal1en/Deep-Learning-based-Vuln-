av_cold void ff_rv34dsp_init(RV34DSPContext *c, DSPContext* dsp) {

    c->rv34_inv_transform    = rv34_inv_transform_noround_c;

    c->rv34_inv_transform_dc = rv34_inv_transform_dc_noround_c;



    c->rv34_idct_add    = rv34_idct_add_c;

    c->rv34_idct_dc_add = rv34_idct_dc_add_c;



    if (HAVE_NEON)

        ff_rv34dsp_init_neon(c, dsp);

    if (ARCH_X86)

        ff_rv34dsp_init_x86(c, dsp);

}
