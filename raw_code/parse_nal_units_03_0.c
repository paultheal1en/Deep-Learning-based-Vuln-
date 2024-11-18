static int parse_nal_units(AVCodecParserContext *s, const uint8_t *buf,

                           int buf_size, AVCodecContext *avctx)

{

    HEVCParserContext *ctx = s->priv_data;

    int ret, i;



    ret = ff_h2645_packet_split(&ctx->pkt, buf, buf_size, avctx, 0, 0,

                                AV_CODEC_ID_HEVC);

    if (ret < 0)

        return ret;



    for (i = 0; i < ctx->pkt.nb_nals; i++) {

        H2645NAL *nal = &ctx->pkt.nals[i];



        /* ignore everything except parameter sets and VCL NALUs */

        switch (nal->type) {

        case NAL_VPS: ff_hevc_decode_nal_vps(&nal->gb, avctx, &ctx->ps);    break;

        case NAL_SPS: ff_hevc_decode_nal_sps(&nal->gb, avctx, &ctx->ps, 1); break;

        case NAL_PPS: ff_hevc_decode_nal_pps(&nal->gb, avctx, &ctx->ps);    break;

        case NAL_TRAIL_R:

        case NAL_TRAIL_N:

        case NAL_TSA_N:

        case NAL_TSA_R:

        case NAL_STSA_N:

        case NAL_STSA_R:

        case NAL_BLA_W_LP:

        case NAL_BLA_W_RADL:

        case NAL_BLA_N_LP:

        case NAL_IDR_W_RADL:

        case NAL_IDR_N_LP:

        case NAL_CRA_NUT:

        case NAL_RADL_N:

        case NAL_RADL_R:

        case NAL_RASL_N:

        case NAL_RASL_R:

            if (buf == avctx->extradata) {

                av_log(avctx, AV_LOG_ERROR, "Invalid NAL unit: %d\n", nal->type);

                return AVERROR_INVALIDDATA;

            }

            hevc_parse_slice_header(s, nal, avctx);

            break;

        }

    }



    return 0;

}
