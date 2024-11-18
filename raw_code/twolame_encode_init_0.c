static av_cold int twolame_encode_init(AVCodecContext *avctx)

{

    TWOLAMEContext *s = avctx->priv_data;

    int ret;



    avctx->frame_size = TWOLAME_SAMPLES_PER_FRAME;

    avctx->delay      = 512 - 32 + 1;



    s->glopts = twolame_init();

    if (!s->glopts)

        return AVERROR(ENOMEM);



    twolame_set_verbosity(s->glopts, s->verbosity);

    twolame_set_mode(s->glopts, s->mode);

    twolame_set_psymodel(s->glopts, s->psymodel);

    twolame_set_energy_levels(s->glopts, s->energy);

    twolame_set_error_protection(s->glopts, s->error_protection);

    twolame_set_copyright(s->glopts, s->copyright);

    twolame_set_original(s->glopts, s->original);



    twolame_set_num_channels(s->glopts, avctx->channels);

    twolame_set_in_samplerate(s->glopts, avctx->sample_rate);

    twolame_set_out_samplerate(s->glopts, avctx->sample_rate);

    if (avctx->flags & CODEC_FLAG_QSCALE || !avctx->bit_rate) {

        twolame_set_VBR(s->glopts, TRUE);

        twolame_set_VBR_level(s->glopts,

                              avctx->global_quality / (float) FF_QP2LAMBDA);

        av_log(avctx, AV_LOG_WARNING,

               "VBR in MP2 is a hack, use another codec that supports it.\n");

    } else {

        twolame_set_bitrate(s->glopts, avctx->bit_rate / 1000);

    }



    ret = twolame_init_params(s->glopts);

    if (ret) {

        twolame_encode_close(avctx);

        return AVERROR_UNKNOWN;

    }



    return 0;

}
