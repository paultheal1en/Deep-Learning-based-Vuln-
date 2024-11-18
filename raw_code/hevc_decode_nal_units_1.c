static int hevc_decode_nal_units(const uint8_t *buf, int buf_size, HEVCParamSets *ps,

                                 int is_nalff, int nal_length_size, void *logctx)

{

    int i;

    int ret = 0;

    H2645Packet pkt = { 0 };



    ret = ff_h2645_packet_split(&pkt, buf, buf_size, logctx, is_nalff, nal_length_size, AV_CODEC_ID_HEVC, 1);

    if (ret < 0) {

        goto done;

    }



    for (i = 0; i < pkt.nb_nals; i++) {

        H2645NAL *nal = &pkt.nals[i];



        /* ignore everything except parameter sets and VCL NALUs */

        switch (nal->type) {

        case HEVC_NAL_VPS: ff_hevc_decode_nal_vps(&nal->gb, logctx, ps);    break;

        case HEVC_NAL_SPS: ff_hevc_decode_nal_sps(&nal->gb, logctx, ps, 1); break;

        case HEVC_NAL_PPS: ff_hevc_decode_nal_pps(&nal->gb, logctx, ps);    break;

        default:

            av_log(logctx, AV_LOG_VERBOSE, "Ignoring NAL type %d in extradata\n", nal->type);

            break;

        }

    }



done:

    ff_h2645_packet_uninit(&pkt);

    return ret;

}
