static av_cold int aac_encode_init(AVCodecContext *avctx)

{

    AACEncContext *s = avctx->priv_data;

    int i;

    const uint8_t *sizes[2];

    uint8_t grouping[AAC_MAX_CHANNELS];

    int lengths[2];



    avctx->frame_size = 1024;



    for (i = 0; i < 16; i++)

        if (avctx->sample_rate == avpriv_mpeg4audio_sample_rates[i])

            break;

    if (i == 16) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported sample rate %d\n", avctx->sample_rate);

        return -1;

    }

    if (avctx->channels > AAC_MAX_CHANNELS) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported number of channels: %d\n", avctx->channels);

        return -1;

    }

    if (avctx->profile != FF_PROFILE_UNKNOWN && avctx->profile != FF_PROFILE_AAC_LOW) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported profile %d\n", avctx->profile);

        return -1;

    }

    if (1024.0 * avctx->bit_rate / avctx->sample_rate > 6144 * avctx->channels) {

        av_log(avctx, AV_LOG_ERROR, "Too many bits per frame requested\n");

        return -1;

    }

    s->samplerate_index = i;



    dsputil_init(&s->dsp, avctx);

    ff_mdct_init(&s->mdct1024, 11, 0, 1.0);

    ff_mdct_init(&s->mdct128,   8, 0, 1.0);

    // window init

    ff_kbd_window_init(ff_aac_kbd_long_1024, 4.0, 1024);

    ff_kbd_window_init(ff_aac_kbd_short_128, 6.0, 128);

    ff_init_ff_sine_windows(10);

    ff_init_ff_sine_windows(7);



    s->chan_map           = aac_chan_configs[avctx->channels-1];

    s->samples            = av_malloc(2 * 1024 * avctx->channels * sizeof(s->samples[0]));

    s->cpe                = av_mallocz(sizeof(ChannelElement) * s->chan_map[0]);

    avctx->extradata      = av_mallocz(5 + FF_INPUT_BUFFER_PADDING_SIZE);

    avctx->extradata_size = 5;

    put_audio_specific_config(avctx);



    sizes[0]   = swb_size_1024[i];

    sizes[1]   = swb_size_128[i];

    lengths[0] = ff_aac_num_swb_1024[i];

    lengths[1] = ff_aac_num_swb_128[i];

    for (i = 0; i < s->chan_map[0]; i++)

        grouping[i] = s->chan_map[i + 1] == TYPE_CPE;

    ff_psy_init(&s->psy, avctx, 2, sizes, lengths, s->chan_map[0], grouping);

    s->psypp = ff_psy_preprocess_init(avctx);

    s->coder = &ff_aac_coders[2];



    s->lambda = avctx->global_quality ? avctx->global_quality : 120;



    ff_aac_tableinit();



    return 0;

}
