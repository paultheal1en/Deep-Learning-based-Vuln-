static void calc_thr_3gpp(const FFPsyWindowInfo *wi, const int num_bands, AacPsyChannel *pch,

                          const uint8_t *band_sizes, const float *coefs)

{

    int i, w, g;

    int start = 0;

    for (w = 0; w < wi->num_windows*16; w += 16) {

        for (g = 0; g < num_bands; g++) {

            AacPsyBand *band = &pch->band[w+g];



            float form_factor = 0.0f;

            float Temp;

            band->energy = 0.0f;

            for (i = 0; i < band_sizes[g]; i++) {

                band->energy += coefs[start+i] * coefs[start+i];

                form_factor  += sqrtf(fabs(coefs[start+i]));

            }

            Temp = band->energy > 0 ? sqrtf((float)band_sizes[g] / band->energy) : 0;

            band->thr      = band->energy * 0.001258925f;

            band->nz_lines = form_factor * sqrtf(Temp);



            start += band_sizes[g];

        }

    }

}
