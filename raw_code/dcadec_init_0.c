static av_cold int dcadec_init(AVCodecContext *avctx)

{

    DCAContext *s = avctx->priv_data;



    s->avctx = avctx;

    s->core.avctx = avctx;

    s->exss.avctx = avctx;

    s->xll.avctx = avctx;

    s->lbr.avctx = avctx;



    ff_dca_init_vlcs();



    if (ff_dca_core_init(&s->core) < 0)

        return AVERROR(ENOMEM);



    if (ff_dca_lbr_init(&s->lbr) < 0)

        return AVERROR(ENOMEM);



    ff_dcadsp_init(&s->dcadsp);

    s->core.dcadsp = &s->dcadsp;

    s->xll.dcadsp = &s->dcadsp;

    s->lbr.dcadsp = &s->dcadsp;



    s->crctab = av_crc_get_table(AV_CRC_16_CCITT);



    switch (avctx->request_channel_layout & ~AV_CH_LAYOUT_NATIVE) {

    case 0:

        s->request_channel_layout = 0;

        break;

    case AV_CH_LAYOUT_STEREO:

    case AV_CH_LAYOUT_STEREO_DOWNMIX:

        s->request_channel_layout = DCA_SPEAKER_LAYOUT_STEREO;

        break;

    case AV_CH_LAYOUT_5POINT0:

        s->request_channel_layout = DCA_SPEAKER_LAYOUT_5POINT0;

        break;

    case AV_CH_LAYOUT_5POINT1:

        s->request_channel_layout = DCA_SPEAKER_LAYOUT_5POINT1;

        break;

    default:

        av_log(avctx, AV_LOG_WARNING, "Invalid request_channel_layout\n");

        break;

    }



    avctx->sample_fmt = AV_SAMPLE_FMT_S32P;

    avctx->bits_per_raw_sample = 24;



    return 0;

}
