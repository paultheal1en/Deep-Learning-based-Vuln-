static av_cold int ac3_encode_init(AVCodecContext *avctx)

{

    int freq = avctx->sample_rate;

    int bitrate = avctx->bit_rate;

    AC3EncodeContext *s = avctx->priv_data;

    int i, j, ch;

    int bw_code;



    avctx->frame_size = AC3_FRAME_SIZE;



    ac3_common_init();



    if (!avctx->channel_layout) {

        av_log(avctx, AV_LOG_WARNING, "No channel layout specified. The "

                                      "encoder will guess the layout, but it "

                                      "might be incorrect.\n");

    }

    if (set_channel_info(s, avctx->channels, &avctx->channel_layout)) {

        av_log(avctx, AV_LOG_ERROR, "invalid channel layout\n");

        return -1;

    }



    /* frequency */

    for (i = 0; i < 3; i++) {

        for (j = 0; j < 3; j++)

            if ((ff_ac3_sample_rate_tab[j] >> i) == freq)

                goto found;

    }

    return -1;

 found:

    s->sample_rate        = freq;

    s->bit_alloc.sr_shift = i;

    s->bit_alloc.sr_code  = j;

    s->bitstream_id       = 8 + s->bit_alloc.sr_shift;

    s->bitstream_mode     = 0; /* complete main audio service */



    /* bitrate & frame size */

    for (i = 0; i < 19; i++) {

        if ((ff_ac3_bitrate_tab[i] >> s->bit_alloc.sr_shift)*1000 == bitrate)

            break;

    }

    if (i == 19)

        return -1;

    s->bit_rate        = bitrate;

    s->frame_size_code = i << 1;

    s->frame_size_min  = 2 * ff_ac3_frame_size_tab[s->frame_size_code][s->bit_alloc.sr_code];

    s->bits_written    = 0;

    s->samples_written = 0;

    s->frame_size      = s->frame_size_min;



    /* set bandwidth */

    if (avctx->cutoff) {

        /* calculate bandwidth based on user-specified cutoff frequency */

        int cutoff     = av_clip(avctx->cutoff, 1, s->sample_rate >> 1);

        int fbw_coeffs = cutoff * 2 * AC3_MAX_COEFS / s->sample_rate;

        bw_code        = av_clip((fbw_coeffs - 73) / 3, 0, 60);

    } else {

        /* use default bandwidth setting */

        /* XXX: should compute the bandwidth according to the frame

           size, so that we avoid annoying high frequency artifacts */

        bw_code = 50;

    }

    for (ch = 0; ch < s->fbw_channels; ch++) {

        /* bandwidth for each channel */

        s->bandwidth_code[ch] = bw_code;

        s->nb_coefs[ch]       = bw_code * 3 + 73;

    }

    if (s->lfe_on)

        s->nb_coefs[s->lfe_channel] = 7; /* LFE channel always has 7 coefs */



    /* initial snr offset */

    s->coarse_snr_offset = 40;



    mdct_init(9);



    avctx->coded_frame= avcodec_alloc_frame();

    avctx->coded_frame->key_frame= 1;



    return 0;

}
