static void decode_scaling_matrices(H264Context *h, SPS *sps,

                                    PPS *pps, int is_sps,

                                    uint8_t(*scaling_matrix4)[16],

                                    uint8_t(*scaling_matrix8)[64])

{

    int fallback_sps = !is_sps && sps->scaling_matrix_present;

    const uint8_t *fallback[4] = {

        fallback_sps ? sps->scaling_matrix4[0] : default_scaling4[0],

        fallback_sps ? sps->scaling_matrix4[3] : default_scaling4[1],

        fallback_sps ? sps->scaling_matrix8[0] : default_scaling8[0],

        fallback_sps ? sps->scaling_matrix8[3] : default_scaling8[1]

    };

    if (get_bits1(&h->gb)) {

        sps->scaling_matrix_present |= is_sps;

        decode_scaling_list(h, scaling_matrix4[0], 16, default_scaling4[0], fallback[0]);        // Intra, Y

        decode_scaling_list(h, scaling_matrix4[1], 16, default_scaling4[0], scaling_matrix4[0]); // Intra, Cr

        decode_scaling_list(h, scaling_matrix4[2], 16, default_scaling4[0], scaling_matrix4[1]); // Intra, Cb

        decode_scaling_list(h, scaling_matrix4[3], 16, default_scaling4[1], fallback[1]);        // Inter, Y

        decode_scaling_list(h, scaling_matrix4[4], 16, default_scaling4[1], scaling_matrix4[3]); // Inter, Cr

        decode_scaling_list(h, scaling_matrix4[5], 16, default_scaling4[1], scaling_matrix4[4]); // Inter, Cb

        if (is_sps || pps->transform_8x8_mode) {

            decode_scaling_list(h, scaling_matrix8[0], 64, default_scaling8[0], fallback[2]); // Intra, Y

            if (sps->chroma_format_idc == 3) {

                decode_scaling_list(h, scaling_matrix8[1], 64, default_scaling8[0], scaling_matrix8[0]); // Intra, Cr

                decode_scaling_list(h, scaling_matrix8[2], 64, default_scaling8[0], scaling_matrix8[1]); // Intra, Cb

            }

            decode_scaling_list(h, scaling_matrix8[3], 64, default_scaling8[1], fallback[3]); // Inter, Y

            if (sps->chroma_format_idc == 3) {

                decode_scaling_list(h, scaling_matrix8[4], 64, default_scaling8[1], scaling_matrix8[3]); // Inter, Cr

                decode_scaling_list(h, scaling_matrix8[5], 64, default_scaling8[1], scaling_matrix8[4]); // Inter, Cb

            }

        }

    }

}
