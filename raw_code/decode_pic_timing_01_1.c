static int decode_pic_timing(HEVCSEIContext *s, GetBitContext *gb, const HEVCParamSets *ps,

                             void *logctx)

{

    HEVCSEIPictureTiming *h = &s->picture_timing;

    HEVCSPS *sps;



    if (!ps->sps_list[s->active_seq_parameter_set_id])

        return(AVERROR(ENOMEM));

    sps = (HEVCSPS*)ps->sps_list[s->active_seq_parameter_set_id]->data;



    if (sps->vui.frame_field_info_present_flag) {

        int pic_struct = get_bits(gb, 4);

        h->picture_struct = AV_PICTURE_STRUCTURE_UNKNOWN;

        if (pic_struct == 2) {

            av_log(logctx, AV_LOG_DEBUG, "BOTTOM Field\n");

            h->picture_struct = AV_PICTURE_STRUCTURE_BOTTOM_FIELD;

        } else if (pic_struct == 1) {

            av_log(logctx, AV_LOG_DEBUG, "TOP Field\n");

            h->picture_struct = AV_PICTURE_STRUCTURE_TOP_FIELD;

        }

        get_bits(gb, 2);                   // source_scan_type

        get_bits(gb, 1);                   // duplicate_flag

    }

    return 1;

}
