void ff_ac3_bit_alloc_calc_psd(int8_t *exp, int start, int end, int16_t *psd,

                               int16_t *band_psd)

{

    int bin, j, k, end1, v;



    /* exponent mapping to PSD */

    for(bin=start;bin<end;bin++) {

        psd[bin]=(3072 - (exp[bin] << 7));

    }



    /* PSD integration */

    j=start;

    k=bin_to_band_tab[start];

    do {

        v = psd[j++];

        end1 = FFMIN(band_start_tab[k+1], end);

        for (; j < end1; j++) {

            /* logadd */

            int adr = FFMIN(FFABS(v - psd[j]) >> 1, 255);

            v = FFMAX(v, psd[j]) + ff_ac3_log_add_tab[adr];

        }

        band_psd[k]=v;

        k++;

    } while (end > band_start_tab[k]);

}
