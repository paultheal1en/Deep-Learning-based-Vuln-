static int64_t find_best_filter(const DCAADPCMEncContext *s, const int32_t *in, int len)

{

    const premultiplied_coeffs *precalc_data = s->private_data;

    int i, j, k = 0;

    int vq;

    int64_t err;

    int64_t min_err = 1ll << 62;

    int64_t corr[15];



    for (i = 0; i <= DCA_ADPCM_COEFFS; i++)

        for (j = i; j <= DCA_ADPCM_COEFFS; j++)

            corr[k++] = calc_corr(in+4, len, i, j);



    for (i = 0; i < DCA_ADPCM_VQCODEBOOK_SZ; i++) {

        err = apply_filter(ff_dca_adpcm_vb[i], corr, *precalc_data);

        if (err < min_err) {

            min_err = err;

            vq = i;

        }

        precalc_data++;

    }



    return vq;

}
