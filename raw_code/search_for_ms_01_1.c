static void search_for_ms(AACEncContext *s, ChannelElement *cpe)

{

    int start = 0, i, w, w2, g, sid_sf_boost;

    float M[128], S[128];

    float *L34 = s->scoefs, *R34 = s->scoefs + 128, *M34 = s->scoefs + 128*2, *S34 = s->scoefs + 128*3;

    const float lambda = s->lambda;

    const float mslambda = FFMIN(1.0f, lambda / 120.f);

    SingleChannelElement *sce0 = &cpe->ch[0];

    SingleChannelElement *sce1 = &cpe->ch[1];

    if (!cpe->common_window)

        return;

    for (w = 0; w < sce0->ics.num_windows; w += sce0->ics.group_len[w]) {

        int min_sf_idx_mid = SCALE_MAX_POS;

        int min_sf_idx_side = SCALE_MAX_POS;

        for (g = 0; g < sce0->ics.num_swb; g++) {

            if (!sce0->zeroes[w*16+g] && sce0->band_type[w*16+g] < RESERVED_BT)

                min_sf_idx_mid = FFMIN(min_sf_idx_mid, sce0->sf_idx[w*16+g]);

            if (!sce1->zeroes[w*16+g] && sce1->band_type[w*16+g] < RESERVED_BT)

                min_sf_idx_side = FFMIN(min_sf_idx_side, sce1->sf_idx[w*16+g]);

        }



        start = 0;

        for (g = 0;  g < sce0->ics.num_swb; g++) {

            float bmax = bval2bmax(g * 17.0f / sce0->ics.num_swb) / 0.0045f;

            cpe->ms_mask[w*16+g] = 0;

            if (!cpe->ch[0].zeroes[w*16+g] && !cpe->ch[1].zeroes[w*16+g]) {

                float Mmax = 0.0f, Smax = 0.0f;



                /* Must compute mid/side SF and book for the whole window group */

                for (w2 = 0; w2 < sce0->ics.group_len[w]; w2++) {

                    for (i = 0; i < sce0->ics.swb_sizes[g]; i++) {

                        M[i] = (sce0->coeffs[start+(w+w2)*128+i]

                              + sce1->coeffs[start+(w+w2)*128+i]) * 0.5;

                        S[i] =  M[i]

                              - sce1->coeffs[start+(w+w2)*128+i];

                    }

                    abs_pow34_v(M34, M, sce0->ics.swb_sizes[g]);

                    abs_pow34_v(S34, S, sce0->ics.swb_sizes[g]);

                    for (i = 0; i < sce0->ics.swb_sizes[g]; i++ ) {

                        Mmax = FFMAX(Mmax, M34[i]);

                        Smax = FFMAX(Smax, S34[i]);

                    }

                }



                for (sid_sf_boost = 0; sid_sf_boost < 4; sid_sf_boost++) {

                    float dist1 = 0.0f, dist2 = 0.0f;

                    int B0 = 0, B1 = 0;

                    int minidx;

                    int mididx, sididx;

                    int midcb, sidcb;



                    minidx = FFMIN(sce0->sf_idx[w*16+g], sce1->sf_idx[w*16+g]);

                    mididx = av_clip(minidx, min_sf_idx_mid, min_sf_idx_mid + SCALE_MAX_DIFF);

                    sididx = av_clip(minidx - sid_sf_boost * 3, min_sf_idx_side, min_sf_idx_side + SCALE_MAX_DIFF);

                    midcb = find_min_book(Mmax, mididx);

                    sidcb = find_min_book(Smax, sididx);



                    if ((mididx > minidx) || (sididx > minidx)) {

                        /* scalefactor range violation, bad stuff, will decrease quality unacceptably */

                        continue;

                    }



                    /* No CB can be zero */

                    midcb = FFMAX(1,midcb);

                    sidcb = FFMAX(1,sidcb);



                    for (w2 = 0; w2 < sce0->ics.group_len[w]; w2++) {

                        FFPsyBand *band0 = &s->psy.ch[s->cur_channel+0].psy_bands[(w+w2)*16+g];

                        FFPsyBand *band1 = &s->psy.ch[s->cur_channel+1].psy_bands[(w+w2)*16+g];

                        float minthr = FFMIN(band0->threshold, band1->threshold);

                        int b1,b2,b3,b4;

                        for (i = 0; i < sce0->ics.swb_sizes[g]; i++) {

                            M[i] = (sce0->coeffs[start+(w+w2)*128+i]

                                  + sce1->coeffs[start+(w+w2)*128+i]) * 0.5;

                            S[i] =  M[i]

                                  - sce1->coeffs[start+(w+w2)*128+i];

                        }



                        abs_pow34_v(L34, sce0->coeffs+start+(w+w2)*128, sce0->ics.swb_sizes[g]);

                        abs_pow34_v(R34, sce1->coeffs+start+(w+w2)*128, sce0->ics.swb_sizes[g]);

                        abs_pow34_v(M34, M,                         sce0->ics.swb_sizes[g]);

                        abs_pow34_v(S34, S,                         sce0->ics.swb_sizes[g]);

                        dist1 += quantize_band_cost(s, &sce0->coeffs[start + (w+w2)*128],

                                                    L34,

                                                    sce0->ics.swb_sizes[g],

                                                    sce0->sf_idx[(w+w2)*16+g],

                                                    sce0->band_type[(w+w2)*16+g],

                                                    lambda / band0->threshold, INFINITY, &b1, NULL, 0);

                        dist1 += quantize_band_cost(s, &sce1->coeffs[start + (w+w2)*128],

                                                    R34,

                                                    sce1->ics.swb_sizes[g],

                                                    sce1->sf_idx[(w+w2)*16+g],

                                                    sce1->band_type[(w+w2)*16+g],

                                                    lambda / band1->threshold, INFINITY, &b2, NULL, 0);

                        dist2 += quantize_band_cost(s, M,

                                                    M34,

                                                    sce0->ics.swb_sizes[g],

                                                    sce0->sf_idx[(w+w2)*16+g],

                                                    sce0->band_type[(w+w2)*16+g],

                                                    lambda / minthr, INFINITY, &b3, NULL, 0);

                        dist2 += quantize_band_cost(s, S,

                                                    S34,

                                                    sce1->ics.swb_sizes[g],

                                                    sce1->sf_idx[(w+w2)*16+g],

                                                    sce1->band_type[(w+w2)*16+g],

                                                    mslambda / (minthr * bmax), INFINITY, &b4, NULL, 0);

                        B0 += b1+b2;

                        B1 += b3+b4;

                        dist1 -= B0;

                        dist2 -= B1;

                    }

                    cpe->ms_mask[w*16+g] = dist2 <= dist1 && B1 < B0;

                    if (cpe->ms_mask[w*16+g]) {

                        /* Setting the M/S mask is useful with I/S or PNS, but only the flag */

                        if (!cpe->is_mask[w*16+g] && sce0->band_type[w*16+g] != NOISE_BT && sce1->band_type[w*16+g] != NOISE_BT) {

                            sce0->sf_idx[w*16+g] = mididx;

                            sce1->sf_idx[w*16+g] = sididx;

                            sce0->band_type[w*16+g] = midcb;

                            sce1->band_type[w*16+g] = sidcb;

                        }

                        break;

                    } else if (B1 > B0) {

                        /* More boost won't fix this */

                        break;

                    }

                }

            }

            start += sce0->ics.swb_sizes[g];

        }

    }

}
