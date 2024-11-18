static void avc_wgt_4width_msa(uint8_t *data,

                               int32_t stride,

                               int32_t height,

                               int32_t log2_denom,

                               int32_t src_weight,

                               int32_t offset_in)

{

    if (2 == height) {

        avc_wgt_4x2_msa(data, stride, log2_denom, src_weight, offset_in);

    } else {

        avc_wgt_4x4multiple_msa(data, stride, height, log2_denom,

                                src_weight, offset_in);

    }

}
