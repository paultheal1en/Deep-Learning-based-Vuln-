void ff_eac3_apply_spectral_extension(AC3DecodeContext *s)

{

    int bin, bnd, ch, i;

    uint8_t wrapflag[SPX_MAX_BANDS]={1,0,}, num_copy_sections, copy_sizes[SPX_MAX_BANDS];

    float rms_energy[SPX_MAX_BANDS];



    /* Set copy index mapping table. Set wrap flags to apply a notch filter at

       wrap points later on. */

    bin = s->spx_dst_start_freq;

    num_copy_sections = 0;

    for (bnd = 0; bnd < s->num_spx_bands; bnd++) {

        int copysize;

        int bandsize = s->spx_band_sizes[bnd];

        if (bin + bandsize > s->spx_src_start_freq) {

            copy_sizes[num_copy_sections++] = bin - s->spx_dst_start_freq;

            bin = s->spx_dst_start_freq;

            wrapflag[bnd] = 1;

        }

        for (i = 0; i < bandsize; i += copysize) {

            if (bin == s->spx_src_start_freq) {

                copy_sizes[num_copy_sections++] = bin - s->spx_dst_start_freq;

                bin = s->spx_dst_start_freq;

            }

            copysize = FFMIN(bandsize - i, s->spx_src_start_freq - bin);

            bin += copysize;

        }

    }

    copy_sizes[num_copy_sections++] = bin - s->spx_dst_start_freq;



    for (ch = 1; ch <= s->fbw_channels; ch++) {

        if (!s->channel_uses_spx[ch])

            continue;



        /* Copy coeffs from normal bands to extension bands */

        bin = s->spx_src_start_freq;

        for (i = 0; i < num_copy_sections; i++) {

            memcpy(&s->transform_coeffs[ch][bin],

                   &s->transform_coeffs[ch][s->spx_dst_start_freq],

                   copy_sizes[i]*sizeof(float));

            bin += copy_sizes[i];

        }



        /* Calculate RMS energy for each SPX band. */

        bin = s->spx_src_start_freq;

        for (bnd = 0; bnd < s->num_spx_bands; bnd++) {

            int bandsize = s->spx_band_sizes[bnd];

            float accum = 0.0f;

            for (i = 0; i < bandsize; i++) {

                float coeff = s->transform_coeffs[ch][bin++];

                accum += coeff * coeff;

            }

            rms_energy[bnd] = sqrtf(accum / bandsize);

        }



        /* Apply a notch filter at transitions between normal and extension

           bands and at all wrap points. */

        if (s->spx_atten_code[ch] >= 0) {

            const float *atten_tab = ff_eac3_spx_atten_tab[s->spx_atten_code[ch]];

            bin = s->spx_src_start_freq - 2;

            for (bnd = 0; bnd < s->num_spx_bands; bnd++) {

                if (wrapflag[bnd]) {

                    float *coeffs = &s->transform_coeffs[ch][bin];

                    coeffs[0] *= atten_tab[0];

                    coeffs[1] *= atten_tab[1];

                    coeffs[2] *= atten_tab[2];

                    coeffs[3] *= atten_tab[1];

                    coeffs[4] *= atten_tab[0];

                }

                bin += s->spx_band_sizes[bnd];

            }

        }



        /* Apply noise-blended coefficient scaling based on previously

           calculated RMS energy, blending factors, and SPX coordinates for

           each band. */

        bin = s->spx_src_start_freq;

        for (bnd = 0; bnd < s->num_spx_bands; bnd++) {

            float nscale = s->spx_noise_blend[ch][bnd] * rms_energy[bnd] * (1.0f / INT32_MIN);

            float sscale = s->spx_signal_blend[ch][bnd];

            for (i = 0; i < s->spx_band_sizes[bnd]; i++) {

                float noise  = nscale * (int32_t)av_lfg_get(&s->dith_state);

                s->transform_coeffs[ch][bin]   *= sscale;

                s->transform_coeffs[ch][bin++] += noise;

            }

        }

    }

}
