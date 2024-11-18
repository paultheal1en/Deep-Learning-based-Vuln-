static av_cold int aacPlus_encode_init(AVCodecContext *avctx)

{

    aacPlusAudioContext *s = avctx->priv_data;

    aacplusEncConfiguration *aacplus_cfg;



    /* number of channels */

    if (avctx->channels < 1 || avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "encoding %d channel(s) is not allowed\n", avctx->channels);

        return -1;

    }



    s->aacplus_handle = aacplusEncOpen(avctx->sample_rate, avctx->channels,

                                       &s->samples_input, &s->max_output_bytes);

    if(!s->aacplus_handle) {

            av_log(avctx, AV_LOG_ERROR, "can't open encoder\n");

            return -1;

    }



    /* check aacplus version */

    aacplus_cfg = aacplusEncGetCurrentConfiguration(s->aacplus_handle);



    /* put the options in the configuration struct */

    if(avctx->profile != FF_PROFILE_AAC_LOW && avctx->profile != FF_PROFILE_UNKNOWN) {

            av_log(avctx, AV_LOG_ERROR, "invalid AAC profile: %d, only LC supported\n", avctx->profile);

            aacplusEncClose(s->aacplus_handle);

            return -1;

    }



    aacplus_cfg->bitRate = avctx->bit_rate;

    aacplus_cfg->bandWidth = avctx->cutoff;

    aacplus_cfg->outputFormat = !(avctx->flags & CODEC_FLAG_GLOBAL_HEADER);

    aacplus_cfg->inputFormat = avctx->sample_fmt == AV_SAMPLE_FMT_FLT ? AACPLUS_INPUT_FLOAT : AACPLUS_INPUT_16BIT;

    if (!aacplusEncSetConfiguration(s->aacplus_handle, aacplus_cfg)) {

        av_log(avctx, AV_LOG_ERROR, "libaacplus doesn't support this output format!\n");

        return -1;

    }



    avctx->frame_size = s->samples_input / avctx->channels;



    /* Set decoder specific info */

    avctx->extradata_size = 0;

    if (avctx->flags & CODEC_FLAG_GLOBAL_HEADER) {



        unsigned char *buffer = NULL;

        unsigned long decoder_specific_info_size;



        if (aacplusEncGetDecoderSpecificInfo(s->aacplus_handle, &buffer,

                                           &decoder_specific_info_size) == 1) {

            avctx->extradata = av_malloc(decoder_specific_info_size + FF_INPUT_BUFFER_PADDING_SIZE);

            avctx->extradata_size = decoder_specific_info_size;

            memcpy(avctx->extradata, buffer, avctx->extradata_size);

        }

        free(buffer);

    }

    return 0;

}
