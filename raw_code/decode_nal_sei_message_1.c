static int decode_nal_sei_message(GetBitContext *gb, void *logctx, HEVCSEI *s,

                                  const HEVCParamSets *ps, int nal_unit_type)

{

    int payload_type = 0;

    int payload_size = 0;

    int byte = 0xFF;

    av_log(logctx, AV_LOG_DEBUG, "Decoding SEI\n");



    while (byte == 0xFF) {



        byte          = get_bits(gb, 8);

        payload_type += byte;

    }

    byte = 0xFF;

    while (byte == 0xFF) {



        byte          = get_bits(gb, 8);

        payload_size += byte;

    }

    if (nal_unit_type == HEVC_NAL_SEI_PREFIX) {

        return decode_nal_sei_prefix(gb, logctx, s, ps, payload_type, payload_size);

    } else { /* nal_unit_type == NAL_SEI_SUFFIX */

        return decode_nal_sei_suffix(gb, logctx, s, payload_type, payload_size);

    }

}