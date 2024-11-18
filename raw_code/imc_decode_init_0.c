static av_cold int imc_decode_init(AVCodecContext *avctx)

{

    int i, j, ret;

    IMCContext *q = avctx->priv_data;

    double r1, r2;



    if ((avctx->codec_id == AV_CODEC_ID_IMC && avctx->channels != 1)

        || (avctx->codec_id == AV_CODEC_ID_IAC && avctx->channels > 2)) {

        av_log_ask_for_sample(avctx, "Number of channels is not supported\n");

        return AVERROR_PATCHWELCOME;

    }



    for (j = 0; j < avctx->channels; j++) {

        q->chctx[j].decoder_reset = 1;



        for (i = 0; i < BANDS; i++)

            q->chctx[j].old_floor[i] = 1.0;



        for (i = 0; i < COEFFS / 2; i++)

            q->chctx[j].last_fft_im[i] = 0;

    }



    /* Build mdct window, a simple sine window normalized with sqrt(2) */

    ff_sine_window_init(q->mdct_sine_window, COEFFS);

    for (i = 0; i < COEFFS; i++)

        q->mdct_sine_window[i] *= sqrt(2.0);

    for (i = 0; i < COEFFS / 2; i++) {

        q->post_cos[i] = (1.0f / 32768) * cos(i / 256.0 * M_PI);

        q->post_sin[i] = (1.0f / 32768) * sin(i / 256.0 * M_PI);



        r1 = sin((i * 4.0 + 1.0) / 1024.0 * M_PI);

        r2 = cos((i * 4.0 + 1.0) / 1024.0 * M_PI);



        if (i & 0x1) {

            q->pre_coef1[i] =  (r1 + r2) * sqrt(2.0);

            q->pre_coef2[i] = -(r1 - r2) * sqrt(2.0);

        } else {

            q->pre_coef1[i] = -(r1 + r2) * sqrt(2.0);

            q->pre_coef2[i] =  (r1 - r2) * sqrt(2.0);

        }

    }



    /* Generate a square root table */



    for (i = 0; i < 30; i++)

        q->sqrt_tab[i] = sqrt(i);



    /* initialize the VLC tables */

    for (i = 0; i < 4 ; i++) {

        for (j = 0; j < 4; j++) {

            huffman_vlc[i][j].table = &vlc_tables[vlc_offsets[i * 4 + j]];

            huffman_vlc[i][j].table_allocated = vlc_offsets[i * 4 + j + 1] - vlc_offsets[i * 4 + j];

            init_vlc(&huffman_vlc[i][j], 9, imc_huffman_sizes[i],

                     imc_huffman_lens[i][j], 1, 1,

                     imc_huffman_bits[i][j], 2, 2, INIT_VLC_USE_NEW_STATIC);

        }

    }



    if (avctx->codec_id == AV_CODEC_ID_IAC) {

        iac_generate_tabs(q, avctx->sample_rate);

    } else {

        memcpy(q->cyclTab,  cyclTab,  sizeof(cyclTab));

        memcpy(q->cyclTab2, cyclTab2, sizeof(cyclTab2));

        memcpy(q->weights1, imc_weights1, sizeof(imc_weights1));

        memcpy(q->weights2, imc_weights2, sizeof(imc_weights2));

    }



    if ((ret = ff_fft_init(&q->fft, 7, 1))) {

        av_log(avctx, AV_LOG_INFO, "FFT init failed\n");

        return ret;

    }

    ff_dsputil_init(&q->dsp, avctx);

    avctx->sample_fmt     = AV_SAMPLE_FMT_FLTP;

    avctx->channel_layout = avctx->channels == 1 ? AV_CH_LAYOUT_MONO

                                                 : AV_CH_LAYOUT_STEREO;



    avcodec_get_frame_defaults(&q->frame);

    avctx->coded_frame = &q->frame;



    return 0;

}
