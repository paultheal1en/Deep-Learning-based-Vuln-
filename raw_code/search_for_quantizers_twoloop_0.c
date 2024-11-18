static void search_for_quantizers_twoloop(AVCodecContext *avctx,

                                          AACEncContext *s,

                                          SingleChannelElement *sce,

                                          const float lambda)

{

    int start = 0, i, w, w2, g;

    int destbits = avctx->bit_rate * 1024.0 / avctx->sample_rate / avctx->channels;

    float dists[128], uplims[128];

    int fflag, minscaler;

    int its  = 0;

    int allz = 0;

    float minthr = INFINITY;



    //XXX: some heuristic to determine initial quantizers will reduce search time

    memset(dists, 0, sizeof(dists));

    //determine zero bands and upper limits

    for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

        for (g = 0;  g < sce->ics.num_swb; g++) {

            int nz = 0;

            float uplim = 0.0f;

            for (w2 = 0; w2 < sce->ics.group_len[w]; w2++) {

                FFPsyBand *band = &s->psy.psy_bands[s->cur_channel*PSY_MAX_BANDS+(w+w2)*16+g];

                uplim += band->threshold;

                if (band->energy <= band->threshold || band->threshold == 0.0f) {

                    sce->zeroes[(w+w2)*16+g] = 1;

                    continue;

                }

                nz = 1;

            }

            uplims[w*16+g] = uplim *512;

            sce->zeroes[w*16+g] = !nz;

            if (nz)

                minthr = FFMIN(minthr, uplim);

            allz = FFMAX(allz, nz);

        }

    }

    for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

        for (g = 0;  g < sce->ics.num_swb; g++) {

            if (sce->zeroes[w*16+g]) {

                sce->sf_idx[w*16+g] = SCALE_ONE_POS;

                continue;

            }

            sce->sf_idx[w*16+g] = SCALE_ONE_POS + FFMIN(log2(uplims[w*16+g]/minthr)*4,59);

        }

    }



    if (!allz)

        return;

    abs_pow34_v(s->scoefs, sce->coeffs, 1024);

    //perform two-loop search

    //outer loop - improve quality

    do {

        int tbits, qstep;

        minscaler = sce->sf_idx[0];

        //inner loop - quantize spectrum to fit into given number of bits

        qstep = its ? 1 : 32;

        do {

            int prev = -1;

            tbits = 0;

            fflag = 0;

            for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

                start = w*128;

                for (g = 0;  g < sce->ics.num_swb; g++) {

                    const float *coefs = sce->coeffs + start;

                    const float *scaled = s->scoefs + start;

                    int bits = 0;

                    int cb;

                    float mindist = INFINITY;

                    int minbits = 0;



                    if (sce->zeroes[w*16+g] || sce->sf_idx[w*16+g] >= 218) {

                        start += sce->ics.swb_sizes[g];

                        continue;

                    }

                    minscaler = FFMIN(minscaler, sce->sf_idx[w*16+g]);

                    for (cb = 0; cb <= ESC_BT; cb++) {

                        float dist = 0.0f;

                        int bb = 0;

                        for (w2 = 0; w2 < sce->ics.group_len[w]; w2++) {

                            int b;

                            dist += quantize_band_cost(s, coefs + w2*128,

                                                       scaled + w2*128,

                                                       sce->ics.swb_sizes[g],

                                                       sce->sf_idx[w*16+g],

                                                       cb,

                                                       lambda,

                                                       INFINITY,

                                                       &b);

                            bb += b;

                        }

                        if (dist < mindist) {

                            mindist = dist;

                            minbits = bb;

                        }

                    }

                    dists[w*16+g] = (mindist - minbits) / lambda;

                    bits = minbits;

                    if (prev != -1) {

                        bits += ff_aac_scalefactor_bits[sce->sf_idx[w*16+g] - prev + SCALE_DIFF_ZERO];

                    }

                    tbits += bits;

                    start += sce->ics.swb_sizes[g];

                    prev = sce->sf_idx[w*16+g];

                }

            }

            if (tbits > destbits) {

                for (i = 0; i < 128; i++)

                    if (sce->sf_idx[i] < 218 - qstep)

                        sce->sf_idx[i] += qstep;

            } else {

                for (i = 0; i < 128; i++)

                    if (sce->sf_idx[i] > 60 - qstep)

                        sce->sf_idx[i] -= qstep;

            }

            qstep >>= 1;

            if (!qstep && tbits > destbits*1.02)

                qstep = 1;

            if (sce->sf_idx[0] >= 217)

                break;

        } while (qstep);



        fflag = 0;

        minscaler = av_clip(minscaler, 60, 255 - SCALE_MAX_DIFF);

        for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

            start = w*128;

            for (g = 0; g < sce->ics.num_swb; g++) {

                int prevsc = sce->sf_idx[w*16+g];

                if (dists[w*16+g] > uplims[w*16+g] && sce->sf_idx[w*16+g] > 60)

                    sce->sf_idx[w*16+g]--;

                sce->sf_idx[w*16+g] = av_clip(sce->sf_idx[w*16+g], minscaler, minscaler + SCALE_MAX_DIFF);

                sce->sf_idx[w*16+g] = FFMIN(sce->sf_idx[w*16+g], 219);

                if (sce->sf_idx[w*16+g] != prevsc)

                    fflag = 1;

            }

        }

        its++;

    } while (fflag && its < 10);

}
