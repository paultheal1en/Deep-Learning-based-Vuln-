static int aac_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                            const AVFrame *frame, int *got_packet_ptr)

{

    AACEncContext *s = avctx->priv_data;

    float **samples = s->planar_samples, *samples2, *la, *overlap;

    ChannelElement *cpe;

    SingleChannelElement *sce;

    int i, its, ch, w, chans, tag, start_ch, ret, frame_bits;

    int ms_mode = 0, is_mode = 0, tns_mode = 0, pred_mode = 0;

    int chan_el_counter[4];

    FFPsyWindowInfo windows[AAC_MAX_CHANNELS];



    if (s->last_frame == 2)

        return 0;



    /* add current frame to queue */

    if (frame) {

        if ((ret = ff_af_queue_add(&s->afq, frame)) < 0)

            return ret;

    }



    copy_input_samples(s, frame);

    if (s->psypp)

        ff_psy_preprocess(s->psypp, s->planar_samples, s->channels);



    if (!avctx->frame_number)

        return 0;



    start_ch = 0;

    for (i = 0; i < s->chan_map[0]; i++) {

        FFPsyWindowInfo* wi = windows + start_ch;

        tag      = s->chan_map[i+1];

        chans    = tag == TYPE_CPE ? 2 : 1;

        cpe      = &s->cpe[i];

        for (ch = 0; ch < chans; ch++) {

            IndividualChannelStream *ics = &cpe->ch[ch].ics;

            int cur_channel = start_ch + ch;

            float clip_avoidance_factor;

            overlap  = &samples[cur_channel][0];

            samples2 = overlap + 1024;

            la       = samples2 + (448+64);

            if (!frame)

                la = NULL;

            if (tag == TYPE_LFE) {

                wi[ch].window_type[0] = ONLY_LONG_SEQUENCE;

                wi[ch].window_shape   = 0;

                wi[ch].num_windows    = 1;

                wi[ch].grouping[0]    = 1;



                /* Only the lowest 12 coefficients are used in a LFE channel.

                 * The expression below results in only the bottom 8 coefficients

                 * being used for 11.025kHz to 16kHz sample rates.

                 */

                ics->num_swb = s->samplerate_index >= 8 ? 1 : 3;

            } else {

                wi[ch] = s->psy.model->window(&s->psy, samples2, la, cur_channel,

                                              ics->window_sequence[0]);

            }

            ics->window_sequence[1] = ics->window_sequence[0];

            ics->window_sequence[0] = wi[ch].window_type[0];

            ics->use_kb_window[1]   = ics->use_kb_window[0];

            ics->use_kb_window[0]   = wi[ch].window_shape;

            ics->num_windows        = wi[ch].num_windows;

            ics->swb_sizes          = s->psy.bands    [ics->num_windows == 8];

            ics->num_swb            = tag == TYPE_LFE ? ics->num_swb : s->psy.num_bands[ics->num_windows == 8];

            ics->swb_offset         = wi[ch].window_type[0] == EIGHT_SHORT_SEQUENCE ?

                                        ff_swb_offset_128 [s->samplerate_index]:

                                        ff_swb_offset_1024[s->samplerate_index];

            ics->tns_max_bands      = wi[ch].window_type[0] == EIGHT_SHORT_SEQUENCE ?

                                        ff_tns_max_bands_128 [s->samplerate_index]:

                                        ff_tns_max_bands_1024[s->samplerate_index];

            clip_avoidance_factor = 0.0f;

            for (w = 0; w < ics->num_windows; w++)

                ics->group_len[w] = wi[ch].grouping[w];

            for (w = 0; w < ics->num_windows; w++) {

                if (wi[ch].clipping[w] > CLIP_AVOIDANCE_FACTOR) {

                    ics->window_clipping[w] = 1;

                    clip_avoidance_factor = FFMAX(clip_avoidance_factor, wi[ch].clipping[w]);

                } else {

                    ics->window_clipping[w] = 0;

                }

            }

            if (clip_avoidance_factor > CLIP_AVOIDANCE_FACTOR) {

                ics->clip_avoidance_factor = CLIP_AVOIDANCE_FACTOR / clip_avoidance_factor;

            } else {

                ics->clip_avoidance_factor = 1.0f;

            }



            apply_window_and_mdct(s, &cpe->ch[ch], overlap);

            if (isnan(cpe->ch->coeffs[0])) {

                av_log(avctx, AV_LOG_ERROR, "Input contains NaN\n");

                return AVERROR(EINVAL);

            }

            avoid_clipping(s, &cpe->ch[ch]);

        }

        start_ch += chans;

    }

    if ((ret = ff_alloc_packet2(avctx, avpkt, 8192 * s->channels, 0)) < 0)

        return ret;

    frame_bits = its = 0;

    do {

        int target_bits, too_many_bits, too_few_bits;



        init_put_bits(&s->pb, avpkt->data, avpkt->size);



        if ((avctx->frame_number & 0xFF)==1 && !(avctx->flags & AV_CODEC_FLAG_BITEXACT))

            put_bitstream_info(s, LIBAVCODEC_IDENT);

        start_ch = 0;

        target_bits = 0;

        memset(chan_el_counter, 0, sizeof(chan_el_counter));

        for (i = 0; i < s->chan_map[0]; i++) {

            FFPsyWindowInfo* wi = windows + start_ch;

            const float *coeffs[2];

            tag      = s->chan_map[i+1];

            chans    = tag == TYPE_CPE ? 2 : 1;

            cpe      = &s->cpe[i];

            cpe->common_window = 0;

            memset(cpe->is_mask, 0, sizeof(cpe->is_mask));

            memset(cpe->ms_mask, 0, sizeof(cpe->ms_mask));

            put_bits(&s->pb, 3, tag);

            put_bits(&s->pb, 4, chan_el_counter[tag]++);

            for (ch = 0; ch < chans; ch++) {

                sce = &cpe->ch[ch];

                coeffs[ch] = sce->coeffs;

                sce->ics.predictor_present = 0;

                memset(&sce->ics.prediction_used, 0, sizeof(sce->ics.prediction_used));

                memset(&sce->tns, 0, sizeof(TemporalNoiseShaping));

                for (w = 0; w < 128; w++)

                    if (sce->band_type[w] > RESERVED_BT)

                        sce->band_type[w] = 0;

            }

            s->psy.bitres.alloc = -1;

            s->psy.bitres.bits = avctx->frame_bits / s->channels;

            s->psy.model->analyze(&s->psy, start_ch, coeffs, wi);

            if (s->psy.bitres.alloc > 0) {

                /* Lambda unused here on purpose, we need to take psy's unscaled allocation */

                target_bits += s->psy.bitres.alloc;

                s->psy.bitres.alloc /= chans;

            }

            s->cur_type = tag;

            for (ch = 0; ch < chans; ch++) {

                s->cur_channel = start_ch + ch;

                s->coder->search_for_quantizers(avctx, s, &cpe->ch[ch], s->lambda);

            }

            if (chans > 1

                && wi[0].window_type[0] == wi[1].window_type[0]

                && wi[0].window_shape   == wi[1].window_shape) {



                cpe->common_window = 1;

                for (w = 0; w < wi[0].num_windows; w++) {

                    if (wi[0].grouping[w] != wi[1].grouping[w]) {

                        cpe->common_window = 0;

                        break;

                    }

                }

            }

            for (ch = 0; ch < chans; ch++) { /* TNS and PNS */

                sce = &cpe->ch[ch];

                s->cur_channel = start_ch + ch;

                if (s->options.pns && s->coder->search_for_pns)

                    s->coder->search_for_pns(s, avctx, sce);

                if (s->options.tns && s->coder->search_for_tns)

                    s->coder->search_for_tns(s, sce);

                if (s->options.tns && s->coder->apply_tns_filt)

                    s->coder->apply_tns_filt(s, sce);

                if (sce->tns.present)

                    tns_mode = 1;

            }

            s->cur_channel = start_ch;

            if (s->options.intensity_stereo) { /* Intensity Stereo */

                if (s->coder->search_for_is)

                    s->coder->search_for_is(s, avctx, cpe);

                if (cpe->is_mode) is_mode = 1;

                apply_intensity_stereo(cpe);

            }

            if (s->options.pred) { /* Prediction */

                for (ch = 0; ch < chans; ch++) {

                    sce = &cpe->ch[ch];

                    s->cur_channel = start_ch + ch;

                    if (s->options.pred && s->coder->search_for_pred)

                        s->coder->search_for_pred(s, sce);

                    if (cpe->ch[ch].ics.predictor_present) pred_mode = 1;

                }

                if (s->coder->adjust_common_prediction)

                    s->coder->adjust_common_prediction(s, cpe);

                for (ch = 0; ch < chans; ch++) {

                    sce = &cpe->ch[ch];

                    s->cur_channel = start_ch + ch;

                    if (s->options.pred && s->coder->apply_main_pred)

                        s->coder->apply_main_pred(s, sce);

                }

                s->cur_channel = start_ch;

            }

            if (s->options.stereo_mode) { /* Mid/Side stereo */

                if (s->options.stereo_mode == -1 && s->coder->search_for_ms)

                    s->coder->search_for_ms(s, cpe);

                else if (cpe->common_window)

                    memset(cpe->ms_mask, 1, sizeof(cpe->ms_mask));

                for (w = 0; w < 128; w++)

                    cpe->ms_mask[w] = cpe->is_mask[w] ? 0 : cpe->ms_mask[w];

                apply_mid_side_stereo(cpe);

            }

            adjust_frame_information(cpe, chans);

            if (chans == 2) {

                put_bits(&s->pb, 1, cpe->common_window);

                if (cpe->common_window) {

                    put_ics_info(s, &cpe->ch[0].ics);

                    if (s->coder->encode_main_pred)

                        s->coder->encode_main_pred(s, &cpe->ch[0]);

                    encode_ms_info(&s->pb, cpe);

                    if (cpe->ms_mode) ms_mode = 1;

                }

            }

            for (ch = 0; ch < chans; ch++) {

                s->cur_channel = start_ch + ch;

                encode_individual_channel(avctx, s, &cpe->ch[ch], cpe->common_window);

            }

            start_ch += chans;

        }



        if (avctx->flags & CODEC_FLAG_QSCALE) {

            /* When using a constant Q-scale, don't mess with lambda */

            break;

        }



        /* rate control stuff

         * target either the nominal bitrate, or what psy's bit reservoir says to target

         * whichever is greatest

         */



        frame_bits = put_bits_count(&s->pb);

        target_bits = FFMAX(target_bits, avctx->bit_rate * 1024 / avctx->sample_rate);

        target_bits = FFMIN(target_bits, 6144 * s->channels - 3);



        /* When using ABR, be strict (but only for increasing) */

        too_many_bits = target_bits + target_bits/2;

        too_few_bits = target_bits - target_bits/8;



        if (   its == 0 /* for steady-state Q-scale tracking */

            || (its < 5 && (frame_bits < too_few_bits || frame_bits > too_many_bits))

            || frame_bits >= 6144 * s->channels - 3  )

        {

            float ratio = ((float)target_bits) / frame_bits;



            if (frame_bits >= too_few_bits && frame_bits <= too_many_bits) {

                /*

                 * This path is for steady-state Q-scale tracking

                 * When frame bits fall within the stable range, we still need to adjust

                 * lambda to maintain it like so in a stable fashion (large jumps in lambda

                 * create artifacts and should be avoided), but slowly

                 */

                ratio = sqrtf(sqrtf(ratio));

                ratio = av_clipf(ratio, 0.9f, 1.1f);

            } else {

                /* Not so fast though */

                ratio = sqrtf(ratio);

            }

            s->lambda = FFMIN(s->lambda * ratio, 65536.f);



            /* Keep iterating if we must reduce and lambda is in the sky */

            if (s->lambda < 300.f || ratio > 0.9f) {

                break;

            } else {

                if (is_mode || ms_mode || tns_mode || pred_mode) {

                    for (i = 0; i < s->chan_map[0]; i++) {

                        // Must restore coeffs

                        chans = tag == TYPE_CPE ? 2 : 1;

                        cpe = &s->cpe[i];

                        for (ch = 0; ch < chans; ch++)

                            memcpy(cpe->ch[ch].coeffs, cpe->ch[ch].pcoeffs, sizeof(cpe->ch[ch].coeffs));

                    }

                }

                its++;

            }

        } else {

            break;

        }

    } while (1);



    put_bits(&s->pb, 3, TYPE_END);

    flush_put_bits(&s->pb);

    avctx->frame_bits = put_bits_count(&s->pb);



    if (!frame)

        s->last_frame++;



    ff_af_queue_remove(&s->afq, avctx->frame_size, &avpkt->pts,

                       &avpkt->duration);



    avpkt->size = put_bits_count(&s->pb) >> 3;

    *got_packet_ptr = 1;

    return 0;

}
