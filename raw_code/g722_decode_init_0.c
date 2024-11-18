static av_cold int g722_decode_init(AVCodecContext * avctx)

{

    G722Context *c = avctx->priv_data;



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "Only mono tracks are allowed.\n");

        return AVERROR_INVALIDDATA;

    }

    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    c->band[0].scale_factor = 8;

    c->band[1].scale_factor = 2;

    c->prev_samples_pos = 22;



    avcodec_get_frame_defaults(&c->frame);

    avctx->coded_frame = &c->frame;



    return 0;

}
