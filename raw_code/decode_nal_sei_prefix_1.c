static int decode_nal_sei_prefix(GetBitContext *gb, HEVCSEIContext *s, const HEVCParamSets *ps,

                                 int type, int size, void *logctx)

{

    switch (type) {

    case 256:  // Mismatched value from HM 8.1

        return decode_nal_sei_decoded_picture_hash(&s->picture_hash, gb);

    case HEVC_SEI_TYPE_FRAME_PACKING:

        return decode_nal_sei_frame_packing_arrangement(&s->frame_packing, gb);

    case HEVC_SEI_TYPE_DISPLAY_ORIENTATION:

        return decode_nal_sei_display_orientation(&s->display_orientation, gb);

    case HEVC_SEI_TYPE_PICTURE_TIMING:

        {

            int ret = decode_pic_timing(s, gb, ps, logctx);

            av_log(logctx, AV_LOG_DEBUG, "Skipped PREFIX SEI %d\n", type);

            skip_bits(gb, 8 * size);

            return ret;

        }

    case HEVC_SEI_TYPE_MASTERING_DISPLAY_INFO:

        return decode_nal_sei_mastering_display_info(&s->mastering_display, gb);

    case HEVC_SEI_TYPE_CONTENT_LIGHT_LEVEL_INFO:

        return decode_nal_sei_content_light_info(&s->content_light, gb);

    case HEVC_SEI_TYPE_ACTIVE_PARAMETER_SETS:

        active_parameter_sets(s, gb, logctx);

        av_log(logctx, AV_LOG_DEBUG, "Skipped PREFIX SEI %d\n", type);

        return 0;

    case HEVC_SEI_TYPE_USER_DATA_REGISTERED_ITU_T_T35:

        return decode_nal_sei_user_data_registered_itu_t_t35(s, gb, size);

    default:

        av_log(logctx, AV_LOG_DEBUG, "Skipped PREFIX SEI %d\n", type);

        skip_bits_long(gb, 8 * size);

        return 0;

    }

}
