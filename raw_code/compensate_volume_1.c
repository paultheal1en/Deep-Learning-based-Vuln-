static int compensate_volume(AVFilterContext *ctx)

{

    struct SOFAlizerContext *s = ctx->priv;

    float compensate;

    float energy = 0;

    float *ir;

    int m;



    if (s->sofa.ncid) {

        /* find IR at front center position in the SOFA file (IR closest to 0°,0°,1m) */

        struct NCSofa *sofa = &s->sofa;

        m = find_m(s, 0, 0, 1);

        /* get energy of that IR and compensate volume */

        ir = sofa->data_ir + 2 * m * sofa->n_samples;

        if (sofa->n_samples & 31) {

            energy = avpriv_scalarproduct_float_c(ir, ir, sofa->n_samples);

        } else {

            energy = s->fdsp->scalarproduct_float(ir, ir, sofa->n_samples);

        }

        compensate = 256 / (sofa->n_samples * sqrt(energy));

        av_log(ctx, AV_LOG_DEBUG, "Compensate-factor: %f\n", compensate);

        ir = sofa->data_ir;

        /* apply volume compensation to IRs */

        s->fdsp->vector_fmul_scalar(ir, ir, compensate, sofa->n_samples * sofa->m_dim * 2);

        emms_c();

    }



    return 0;

}
