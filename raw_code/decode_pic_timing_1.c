static int decode_pic_timing(HEVCContext *s)

{

    GetBitContext *gb = &s->HEVClc->gb;

    HEVCSPS *sps = (HEVCSPS*)s->sps_list[s->active_seq_parameter_set_id]->data;



    if (!sps)

        return(AVERROR(ENOMEM));



    if (sps->vui.frame_field_info_present_flag) {

        int pic_struct = get_bits(gb, 4);

        s->picture_struct = AV_PICTURE_STRUCTURE_UNKNOWN;

        if (pic_struct == 2) {

            av_log(s->avctx, AV_LOG_DEBUG, "BOTTOM Field\n");

            s->picture_struct = AV_PICTURE_STRUCTURE_BOTTOM_FIELD;

        } else if (pic_struct == 1) {

            av_log(s->avctx, AV_LOG_DEBUG, "TOP Field\n");

            s->picture_struct = AV_PICTURE_STRUCTURE_TOP_FIELD;

        }

        get_bits(gb, 2);                   // source_scan_type

        get_bits(gb, 1);                   // duplicate_flag

    }

    return 1;

}
