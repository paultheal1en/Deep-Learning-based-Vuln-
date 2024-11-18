static int decode_block(BinkAudioContext *s, float **out, int use_dct)

{

    int ch, i, j, k;

    float q, quant[25];

    int width, coeff;

    GetBitContext *gb = &s->gb;



    if (use_dct)

        skip_bits(gb, 2);



    for (ch = 0; ch < s->channels; ch++) {

        FFTSample *coeffs = out[ch];



        if (s->version_b) {

            if (get_bits_left(gb) < 64)

                return AVERROR_INVALIDDATA;

            coeffs[0] = av_int2float(get_bits_long(gb, 32)) * s->root;

            coeffs[1] = av_int2float(get_bits_long(gb, 32)) * s->root;

        } else {

            if (get_bits_left(gb) < 58)

                return AVERROR_INVALIDDATA;

            coeffs[0] = get_float(gb) * s->root;

            coeffs[1] = get_float(gb) * s->root;

        }



        if (get_bits_left(gb) < s->num_bands * 8)

            return AVERROR_INVALIDDATA;

        for (i = 0; i < s->num_bands; i++) {

            int value = get_bits(gb, 8);

            quant[i]  = quant_table[FFMIN(value, 95)];

        }



        k = 0;

        q = quant[0];



        // parse coefficients

        i = 2;

        while (i < s->frame_len) {

            if (s->version_b) {

                j = i + 16;

            } else {

                int v;

                GET_BITS_SAFE(v, 1);

                if (v) {

                    GET_BITS_SAFE(v, 4);

                    j = i + rle_length_tab[v] * 8;

                } else {

                    j = i + 8;

                }

            }



            j = FFMIN(j, s->frame_len);



            GET_BITS_SAFE(width, 4);

            if (width == 0) {

                memset(coeffs + i, 0, (j - i) * sizeof(*coeffs));

                i = j;

                while (s->bands[k] < i)

                    q = quant[k++];

            } else {

                while (i < j) {

                    if (s->bands[k] == i)

                        q = quant[k++];

                    GET_BITS_SAFE(coeff, width);

                    if (coeff) {

                        int v;

                        GET_BITS_SAFE(v, 1);

                        if (v)

                            coeffs[i] = -q * coeff;

                        else

                            coeffs[i] =  q * coeff;

                    } else {

                        coeffs[i] = 0.0f;

                    }

                    i++;

                }

            }

        }



        if (CONFIG_BINKAUDIO_DCT_DECODER && use_dct) {

            coeffs[0] /= 0.5;

            s->trans.dct.dct_calc(&s->trans.dct,  coeffs);

        }

        else if (CONFIG_BINKAUDIO_RDFT_DECODER)

            s->trans.rdft.rdft_calc(&s->trans.rdft, coeffs);

    }



    for (ch = 0; ch < s->channels; ch++) {

        int j;

        int count = s->overlap_len * s->channels;

        if (!s->first) {

            j = ch;

            for (i = 0; i < s->overlap_len; i++, j += s->channels)

                out[ch][i] = (s->previous[ch][i] * (count - j) +

                                      out[ch][i] *          j) / count;

        }

        memcpy(s->previous[ch], &out[ch][s->frame_len - s->overlap_len],

               s->overlap_len * sizeof(*s->previous[ch]));

    }



    s->first = 0;



    return 0;

}
