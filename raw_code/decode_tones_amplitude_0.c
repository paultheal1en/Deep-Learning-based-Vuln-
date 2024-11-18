static void decode_tones_amplitude(GetBitContext *gb, Atrac3pChanUnitCtx *ctx,

                                   int ch_num, int band_has_tones[])

{

    int mode, sb, j, i, diff, maxdiff, fi, delta, pred;

    Atrac3pWaveParam *wsrc, *wref;

    int refwaves[48];

    Atrac3pWavesData *dst = ctx->channels[ch_num].tones_info;

    Atrac3pWavesData *ref = ctx->channels[0].tones_info;



    if (ch_num) {

        for (sb = 0; sb < ctx->waves_info->num_tone_bands; sb++) {

            if (!band_has_tones[sb] || !dst[sb].num_wavs)

                continue;

            wsrc = &ctx->waves_info->waves[dst[sb].start_index];

            wref = &ctx->waves_info->waves[ref[sb].start_index];

            for (j = 0; j < dst[sb].num_wavs; j++) {

                for (i = 0, fi = 0, maxdiff = 1024; i < ref[sb].num_wavs; i++) {

                    diff = FFABS(wsrc[j].freq_index - wref[i].freq_index);

                    if (diff < maxdiff) {

                        maxdiff = diff;

                        fi      = i;

                    }

                }



                if (maxdiff < 8)

                    refwaves[dst[sb].start_index + j] = fi + ref[sb].start_index;

                else if (j < ref[sb].num_wavs)

                    refwaves[dst[sb].start_index + j] = j + ref[sb].start_index;

                else

                    refwaves[dst[sb].start_index + j] = -1;

            }

        }

    }



    mode = get_bits(gb, ch_num + 1);



    switch (mode) {

    case 0: /** fixed-length coding */

        for (sb = 0; sb < ctx->waves_info->num_tone_bands; sb++) {

            if (!band_has_tones[sb] || !dst[sb].num_wavs)

                continue;

            if (ctx->waves_info->amplitude_mode)

                for (i = 0; i < dst[sb].num_wavs; i++)

                    ctx->waves_info->waves[dst[sb].start_index + i].amp_sf = get_bits(gb, 6);

            else

                ctx->waves_info->waves[dst[sb].start_index].amp_sf = get_bits(gb, 6);

        }

        break;

    case 1: /** min + VLC delta */

        for (sb = 0; sb < ctx->waves_info->num_tone_bands; sb++) {

            if (!band_has_tones[sb] || !dst[sb].num_wavs)

                continue;

            if (ctx->waves_info->amplitude_mode)

                for (i = 0; i < dst[sb].num_wavs; i++)

                    ctx->waves_info->waves[dst[sb].start_index + i].amp_sf =

                        get_vlc2(gb, tone_vlc_tabs[3].table,

                                 tone_vlc_tabs[3].bits, 1) + 20;

            else

                ctx->waves_info->waves[dst[sb].start_index].amp_sf =

                    get_vlc2(gb, tone_vlc_tabs[4].table,

                             tone_vlc_tabs[4].bits, 1) + 24;

        }

        break;

    case 2: /** VLC modulo delta to master (slave only) */

        for (sb = 0; sb < ctx->waves_info->num_tone_bands; sb++) {

            if (!band_has_tones[sb] || !dst[sb].num_wavs)

                continue;

            for (i = 0; i < dst[sb].num_wavs; i++) {

                delta = get_vlc2(gb, tone_vlc_tabs[5].table,

                                 tone_vlc_tabs[5].bits, 1);

                delta = sign_extend(delta, 5);

                pred  = refwaves[dst[sb].start_index + i] >= 0 ?

                        ctx->waves_info->waves[refwaves[dst[sb].start_index + i]].amp_sf : 34;

                ctx->waves_info->waves[dst[sb].start_index + i].amp_sf = (pred + delta) & 0x3F;

            }

        }

        break;

    case 3: /** clone master (slave only) */

        for (sb = 0; sb < ctx->waves_info->num_tone_bands; sb++) {

            if (!band_has_tones[sb])

                continue;

            for (i = 0; i < dst[sb].num_wavs; i++)

                ctx->waves_info->waves[dst[sb].start_index + i].amp_sf =

                    refwaves[dst[sb].start_index + i] >= 0

                    ? ctx->waves_info->waves[refwaves[dst[sb].start_index + i]].amp_sf

                    : 32;

        }

        break;

    }

}
