static int decode_buffering_period(H264Context *h)

{

    unsigned int sps_id;

    int sched_sel_idx;

    SPS *sps;



    sps_id = get_ue_golomb_31(&h->gb);

    if (sps_id > 31 || !h->sps_buffers[sps_id]) {

        av_log(h->avctx, AV_LOG_ERROR,

               "non-existing SPS %d referenced in buffering period\n", sps_id);

        return AVERROR_INVALIDDATA;

    }

    sps = h->sps_buffers[sps_id];



    // NOTE: This is really so duplicated in the standard... See H.264, D.1.1

    if (sps->nal_hrd_parameters_present_flag) {

        for (sched_sel_idx = 0; sched_sel_idx < sps->cpb_cnt; sched_sel_idx++) {

            h->initial_cpb_removal_delay[sched_sel_idx] =

                get_bits(&h->gb, sps->initial_cpb_removal_delay_length);

            // initial_cpb_removal_delay_offset

            skip_bits(&h->gb, sps->initial_cpb_removal_delay_length);

        }

    }

    if (sps->vcl_hrd_parameters_present_flag) {

        for (sched_sel_idx = 0; sched_sel_idx < sps->cpb_cnt; sched_sel_idx++) {

            h->initial_cpb_removal_delay[sched_sel_idx] =

                get_bits(&h->gb, sps->initial_cpb_removal_delay_length);

            // initial_cpb_removal_delay_offset

            skip_bits(&h->gb, sps->initial_cpb_removal_delay_length);

        }

    }



    h->sei_buffering_period_present = 1;

    return 0;

}
