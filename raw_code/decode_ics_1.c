static int decode_ics(AACContext *ac, SingleChannelElement *sce,

                      GetBitContext *gb, int common_window, int scale_flag)

{

    Pulse pulse;

    TemporalNoiseShaping    *tns = &sce->tns;

    IndividualChannelStream *ics = &sce->ics;

    INTFLOAT *out = sce->coeffs;

    int global_gain, eld_syntax, er_syntax, pulse_present = 0;

    int ret;



    eld_syntax = ac->oc[1].m4ac.object_type == AOT_ER_AAC_ELD;

    er_syntax  = ac->oc[1].m4ac.object_type == AOT_ER_AAC_LC ||

                 ac->oc[1].m4ac.object_type == AOT_ER_AAC_LTP ||

                 ac->oc[1].m4ac.object_type == AOT_ER_AAC_LD ||

                 ac->oc[1].m4ac.object_type == AOT_ER_AAC_ELD;



    /* This assignment is to silence a GCC warning about the variable being used

     * uninitialized when in fact it always is.

     */

    pulse.num_pulse = 0;



    global_gain = get_bits(gb, 8);



    if (!common_window && !scale_flag) {

        if (decode_ics_info(ac, ics, gb) < 0)

            return AVERROR_INVALIDDATA;

    }



    if ((ret = decode_band_types(ac, sce->band_type,

                                 sce->band_type_run_end, gb, ics)) < 0)

        return ret;

    if ((ret = decode_scalefactors(ac, sce->sf, gb, global_gain, ics,

                                  sce->band_type, sce->band_type_run_end)) < 0)

        return ret;



    pulse_present = 0;

    if (!scale_flag) {

        if (!eld_syntax && (pulse_present = get_bits1(gb))) {

            if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {

                av_log(ac->avctx, AV_LOG_ERROR,

                       "Pulse tool not allowed in eight short sequence.\n");

                return AVERROR_INVALIDDATA;

            }

            if (decode_pulses(&pulse, gb, ics->swb_offset, ics->num_swb)) {

                av_log(ac->avctx, AV_LOG_ERROR,

                       "Pulse data corrupt or invalid.\n");

                return AVERROR_INVALIDDATA;

            }

        }

        tns->present = get_bits1(gb);

        if (tns->present && !er_syntax)

            if (decode_tns(ac, tns, gb, ics) < 0)

                return AVERROR_INVALIDDATA;

        if (!eld_syntax && get_bits1(gb)) {

            avpriv_request_sample(ac->avctx, "SSR");

            return AVERROR_PATCHWELCOME;

        }

        // I see no textual basis in the spec for this occurring after SSR gain

        // control, but this is what both reference and real implmentations do

        if (tns->present && er_syntax)

            if (decode_tns(ac, tns, gb, ics) < 0)

                return AVERROR_INVALIDDATA;

    }



    if (decode_spectrum_and_dequant(ac, out, gb, sce->sf, pulse_present,

                                    &pulse, ics, sce->band_type) < 0)

        return AVERROR_INVALIDDATA;



    if (ac->oc[1].m4ac.object_type == AOT_AAC_MAIN && !common_window)

        apply_prediction(ac, sce);



    return 0;

}
