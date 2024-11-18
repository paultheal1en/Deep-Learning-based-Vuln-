static int ac3_decode_frame(AVCodecContext * avctx, void *data, int *data_size, uint8_t *buf, int buf_size)

{

    AC3DecodeContext *ctx = (AC3DecodeContext *)avctx->priv_data;

    ac3_audio_block *ab = &ctx->audio_block;

    int frame_start;

    int i, j, k, l, value;

    float tmp_block_first_half[128], tmp_block_second_half[128];

    int16_t *out_samples = (int16_t *)data;

    int nfchans;



    //Synchronize the frame.

    frame_start = ac3_synchronize(buf, buf_size);

    if (frame_start == -1) {

        av_log(avctx, AV_LOG_ERROR, "frame is not synchronized\n");

        *data_size = 0;

        return buf_size;

    }



    //Initialize the GetBitContext with the start of valid AC3 Frame.

    init_get_bits(&(ctx->gb), buf + frame_start, (buf_size - frame_start) * 8);



    //Parse the syncinfo.

    //If 'fscod' or 'bsid' is not valid the decoder shall mute as per the standard.

    if (!ac3_parse_sync_info(ctx)) {

        av_log(avctx, AV_LOG_ERROR, "\n");

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



    for (i = 0; i < MAX_BLOCKS; i++)

        memset(ctx->delay[i], 0, sizeof(ctx->delay[i]));

    avctx->sample_rate = ctx->sync_info.sampling_rate;

    avctx->bit_rate = ctx->sync_info.bit_rate;

    if (avctx->channels == 0) {

        //avctx->channels = ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0);

        ctx->output = AC3_OUTPUT_UNMODIFIED;

    }

    else if ((ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0)) < avctx->channels) {

        av_log(avctx, AV_LOG_INFO, "ac3_decoder: AC3 Source Channels Are Less Then Specified %d: Output to %d Channels\n",

                avctx->channels, (ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0)));

        //avctx->channels = ctx->bsi.nfchans + ((ctx->bsi.flags & AC3_BSI_LFEON) ? 1 : 0);

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





    av_log(avctx, AV_LOG_INFO, "channels = %d \t bit rate = %d \t sampling rate = %d \n", avctx->channels, avctx->sample_rate, avctx->bit_rate);



    //Parse the Audio Blocks.

    *data_size = 0;

    for (i = 0; i < 6; i++) {

        if (ac3_parse_audio_block(ctx, i)) {

            av_log(avctx, AV_LOG_ERROR, "error parsing the audio block\n");

            *data_size = 0;

            return -1;

        }



        av_log(NULL, AV_LOG_INFO, "doing imdct\n");



        if (ctx->bsi.flags & AC3_BSI_LFEON) {

            ff_imdct_calc(&ctx->imdct_ctx_512, ctx->tmp_output, ab->transform_coeffs[0], ctx->tmp_imdct);

            for (l = 0; l < 256; l++)

                ab->block_output[0][l] = ctx->tmp_output[l] * window[l] + ctx->delay[0][l] * window[255 -l];

            memcpy(ctx->delay[0], ctx->tmp_output + 256, sizeof(ctx->delay[0]));

        }



        for (j = 0; j < ctx->bsi.nfchans; j++) {

            if (ctx->audio_block.blksw & (1 << j)) {

                for (k = 0; k < 128; k++) {

                    tmp_block_first_half[k] = ab->transform_coeffs[j + 1][2 * k];

                    tmp_block_second_half[k] = ab->transform_coeffs[j + 1][2 * k + 1];

                }

                ff_imdct_calc(&ctx->imdct_ctx_256, ctx->tmp_output, tmp_block_first_half, ctx->tmp_imdct);

                for (l = 0; l < 256; l++)

                    ab->block_output[j + 1][l] = ctx->tmp_output[l] * window[l] + ctx->delay[j + 1][l] * window[255 - l];

                ff_imdct_calc(&ctx->imdct_ctx_256, ctx->delay[j + 1], tmp_block_second_half, ctx->tmp_imdct);

            } else {

                ff_imdct_calc(&ctx->imdct_ctx_512, ctx->tmp_output, ab->transform_coeffs[j + 1], ctx->tmp_imdct);

                for (l = 0; l < 256; l++)

                    ab->block_output[j + 1][l] = ctx->tmp_output[l] * window[l] + ctx->delay[j + 1][l] * window[255 - l];

                memcpy(ctx->delay[j + 1], ctx->tmp_output + 256, sizeof(ctx->delay[j + 1]));

            }

        }

        if (ctx->bsi.flags & AC3_BSI_LFEON) {

            for (l = 0; l < 256; l++) {

                value = lrint(ab->block_output[0][l]);

                if (value < -32768)

                    value = -32768;

                else if (value > 32767)

                    value = 32767;

                *(out_samples++) = value;

            }

            *data_size += 256 * sizeof(int16_t);

        }

        do_downmix(ctx);

        if (ctx->output == AC3_OUTPUT_UNMODIFIED)

            nfchans = ctx->bsi.nfchans;

        else

            nfchans = avctx->channels;

        for (k = 0; k < nfchans; k++)

            for (l = 0; l < 256; l++) {

                value = lrint(ab->block_output[k + 1][l]);

                if (value < -32768)

                    value = -32768;

                else if (value > 32767)

                    value = 32767;

                *(out_samples++) = value;

            }

        *data_size += nfchans * 256 * sizeof (int16_t);

    }



    return ctx->sync_info.framesize;

}
