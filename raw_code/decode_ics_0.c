static int decode_ics(AACContext * ac, SingleChannelElement * sce, GetBitContext * gb, int common_window, int scale_flag) {

    Pulse pulse;

    TemporalNoiseShaping * tns = &sce->tns;

    IndividualChannelStream * ics = &sce->ics;

    float * out = sce->coeffs;

    int global_gain, pulse_present = 0;



    /* This assignment is to silence a GCC warning about the variable being used

     * uninitialized when in fact it always is.

     */

    pulse.num_pulse = 0;



    global_gain = get_bits(gb, 8);



    if (!common_window && !scale_flag) {

        if (decode_ics_info(ac, ics, gb, 0) < 0)

            return -1;

    }



    if (decode_band_types(ac, sce->band_type, sce->band_type_run_end, gb, ics) < 0)

        return -1;

    if (decode_scalefactors(ac, sce->sf, gb, global_gain, ics, sce->band_type, sce->band_type_run_end) < 0)

        return -1;



    pulse_present = 0;

    if (!scale_flag) {

        if ((pulse_present = get_bits1(gb))) {

            if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {

                av_log(ac->avccontext, AV_LOG_ERROR, "Pulse tool not allowed in eight short sequence.\n");

                return -1;

            }

            decode_pulses(&pulse, gb, ics->swb_offset);

        }

        if ((tns->present = get_bits1(gb)) && decode_tns(ac, tns, gb, ics))

            return -1;

        if (get_bits1(gb)) {

            av_log_missing_feature(ac->avccontext, "SSR", 1);

            return -1;

        }

    }



    if (decode_spectrum_and_dequant(ac, out, gb, sce->sf, pulse_present, &pulse, ics, sce->band_type) < 0)

        return -1;

    return 0;

}
