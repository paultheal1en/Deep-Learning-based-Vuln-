static int ac3_decode_frame(AVCodecContext * avctx, void *data, int *data_size, uint8_t * buf, int buf_size)

{

    AC3DecodeContext *ctx = avctx->priv_data;

    int frame_start;

    int i, j, k, l;

    float tmp0[128], tmp1[128], tmp[512];

    short *out_samples = (short *)data;

    float *samples = ctx->samples;



    //Synchronize the frame.

    frame_start = ac3_synchronize(buf, buf_size);

    if (frame_start == -1) {

        av_log(avctx, AV_LOG_ERROR, "frame is not synchronized\n");

        *data_size = 0;

        return -1;

    }



    //Initialize the GetBitContext with the start of valid AC3 Frame.

    init_get_bits(&(ctx->gb), buf + frame_start, (buf_size - frame_start) * 8);

    //Parse the syncinfo.

    ////If 'fscod' is not valid the decoder shall mute as per the standard.

    if (ac3_parse_sync_info(ctx)) {

        av_log(avctx, AV_LOG_ERROR, "fscod is not valid\n");

        *data_size = 0;

        return -1;

    }



    //Check for the errors.

    /* if (ac3_error_check(ctx)) {

        *data_size = 0;

        return -1;

    } */



    //Parse the BSI.

    //If 'bsid' is not valid decoder shall not decode the audio as per the standard.

    if (ac3_parse_bsi(ctx)) {

        av_log(avctx, AV_LOG_ERROR, "bsid is not valid\n");

        *data_size = 0;

        return -1;

    }



    avctx->sample_rate = ctx->sync_info.sampling_rate;

    if (avctx->channels == 0) {

        avctx->channels = ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0);

        ctx->output = AC3_OUTPUT_UNMODIFIED;

    }

    else if ((ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0)) < avctx->channels) {

        av_log(avctx, AV_LOG_INFO, "ac3_decoder: AC3 Source Channels Are Less Then Specified %d: Output to %d Channels\n",

                avctx->channels, (ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0)));

        avctx->channels = ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0);

        ctx->output = AC3_OUTPUT_UNMODIFIED;

    }

    else if (avctx->channels == 1) {

        ctx->output = AC3_OUTPUT_MONO;

    } else if (avctx->channels == 2) {

        if (ctx->bsi.dsurmod == 0x02)

            ctx->output = AC3_OUTPUT_DOLBY;

        else

            ctx->output = AC3_OUTPUT_STEREO;

    }





    avctx->bit_rate = ctx->sync_info.bit_rate;

    av_log(avctx, AV_LOG_INFO, "channels = %d \t bit rate = %d \t sampling rate = %d \n", avctx->channels, avctx->sample_rate, avctx->bit_rate);



    //Parse the Audio Blocks.

    for (i = 0; i < 6; i++) {

        if (ac3_parse_audio_block(ctx, i)) {

            av_log(avctx, AV_LOG_ERROR, "error parsing the audio block\n");

            *data_size = 0;

            return -1;

        }

        samples = ctx->samples;

        if (ctx->bsi.flags & AC3_BSI_LFEON) {

            ff_imdct_calc(&ctx->imdct_ctx_512, ctx->samples + 1536, samples, tmp);

            for (l = 0; l < 256; l++)

                samples[l] = (ctx->samples + 1536)[l];

            float_to_int(samples, out_samples, 256);

            samples += 256;

            out_samples += 256;

        }

        for (j = 0; j < ctx->bsi.nfchans; j++) {

            if (ctx->audio_block.blksw & (1 << j)) {

                for (k = 0; k < 128; k++) {

                    tmp0[k] = samples[2 * k];

                    tmp1[k] = samples[2 * k + 1];

                }

                ff_imdct_calc(&ctx->imdct_ctx_256, ctx->samples + 1536, tmp0, tmp);

                for (l = 0; l < 256; l++)

                    samples[l] = (ctx->samples + 1536)[l] * window[l] + (ctx->samples + 2048)[l] * window[255 - l];

                ff_imdct_calc(&ctx->imdct_ctx_256, ctx->samples + 2048, tmp1, tmp);

                float_to_int(samples, out_samples, 256);

                samples += 256;

                out_samples += 256;

            }

            else {

                ff_imdct_calc(&ctx->imdct_ctx_512, ctx->samples + 1536, samples, tmp);

                for (l = 0; l < 256; l++)

                    samples[l] = (ctx->samples + 1536)[l] * window[l] + (ctx->samples + 2048)[l] * window[255 - l];

                float_to_int(samples, out_samples, 256);

                memcpy(ctx->samples + 2048, ctx->samples + 1792, 256 * sizeof (float));

                samples += 256;

                out_samples += 256;

            }

        }

    }

    *data_size = 6 * ctx->bsi.nfchans * 256 * sizeof (int16_t);



    return (buf_size - frame_start);

}
