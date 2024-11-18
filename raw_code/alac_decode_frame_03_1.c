static int alac_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame_ptr, AVPacket *avpkt)

{

    const uint8_t *inbuffer = avpkt->data;

    int input_buffer_size = avpkt->size;

    ALACContext *alac = avctx->priv_data;



    int channels;

    unsigned int outputsamples;

    int hassize;

    unsigned int readsamplesize;

    int isnotcompressed;

    uint8_t interlacing_shift;

    uint8_t interlacing_leftweight;

    int i, ch, ret;



    init_get_bits(&alac->gb, inbuffer, input_buffer_size * 8);



    channels = get_bits(&alac->gb, 3) + 1;

    if (channels != avctx->channels) {

        av_log(avctx, AV_LOG_ERROR, "frame header channel count mismatch\n");

        return AVERROR_INVALIDDATA;

    }



    /* 2^result = something to do with output waiting.

     * perhaps matters if we read > 1 frame in a pass?

     */

    skip_bits(&alac->gb, 4);



    skip_bits(&alac->gb, 12); /* unknown, skip 12 bits */



    /* the output sample size is stored soon */

    hassize = get_bits1(&alac->gb);



    alac->extra_bits = get_bits(&alac->gb, 2) << 3;



    /* whether the frame is compressed */

    isnotcompressed = get_bits1(&alac->gb);



    if (hassize) {

        /* now read the number of samples as a 32bit integer */

        outputsamples = get_bits_long(&alac->gb, 32);

        if(outputsamples > alac->setinfo_max_samples_per_frame){

            av_log(avctx, AV_LOG_ERROR, "outputsamples %d > %d\n", outputsamples, alac->setinfo_max_samples_per_frame);

            return -1;

        }

    } else

        outputsamples = alac->setinfo_max_samples_per_frame;



    /* get output buffer */

    if (outputsamples > INT32_MAX) {

        av_log(avctx, AV_LOG_ERROR, "unsupported block size: %u\n", outputsamples);

        return AVERROR_INVALIDDATA;

    }

    alac->frame.nb_samples = outputsamples;

    if ((ret = avctx->get_buffer(avctx, &alac->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    readsamplesize = alac->setinfo_sample_size - alac->extra_bits + channels - 1;

    if (readsamplesize > MIN_CACHE_BITS) {

        av_log(avctx, AV_LOG_ERROR, "readsamplesize too big (%d)\n", readsamplesize);

        return -1;

    }



    if (!isnotcompressed) {

        /* so it is compressed */

        int16_t predictor_coef_table[MAX_CHANNELS][32];

        int predictor_coef_num[MAX_CHANNELS];

        int prediction_type[MAX_CHANNELS];

        int prediction_quantitization[MAX_CHANNELS];

        int ricemodifier[MAX_CHANNELS];



        interlacing_shift = get_bits(&alac->gb, 8);

        interlacing_leftweight = get_bits(&alac->gb, 8);



        for (ch = 0; ch < channels; ch++) {

            prediction_type[ch] = get_bits(&alac->gb, 4);

            prediction_quantitization[ch] = get_bits(&alac->gb, 4);



            ricemodifier[ch] = get_bits(&alac->gb, 3);

            predictor_coef_num[ch] = get_bits(&alac->gb, 5);



            /* read the predictor table */

            for (i = 0; i < predictor_coef_num[ch]; i++)

                predictor_coef_table[ch][i] = (int16_t)get_bits(&alac->gb, 16);

        }



        if (alac->extra_bits) {

            for (i = 0; i < outputsamples; i++) {

                for (ch = 0; ch < channels; ch++)

                    alac->extra_bits_buffer[ch][i] = get_bits(&alac->gb, alac->extra_bits);

            }

        }

        for (ch = 0; ch < channels; ch++) {

            bastardized_rice_decompress(alac,

                                        alac->predicterror_buffer[ch],

                                        outputsamples,

                                        readsamplesize,

                                        alac->setinfo_rice_initialhistory,

                                        alac->setinfo_rice_kmodifier,

                                        ricemodifier[ch] * alac->setinfo_rice_historymult / 4,

                                        (1 << alac->setinfo_rice_kmodifier) - 1);



            if (prediction_type[ch] == 0) {

                /* adaptive fir */

                predictor_decompress_fir_adapt(alac->predicterror_buffer[ch],

                                               alac->outputsamples_buffer[ch],

                                               outputsamples,

                                               readsamplesize,

                                               predictor_coef_table[ch],

                                               predictor_coef_num[ch],

                                               prediction_quantitization[ch]);

            } else {

                av_log(avctx, AV_LOG_ERROR, "FIXME: unhandled prediction type: %i\n", prediction_type[ch]);

                /* I think the only other prediction type (or perhaps this is

                 * just a boolean?) runs adaptive fir twice.. like:

                 * predictor_decompress_fir_adapt(predictor_error, tempout, ...)

                 * predictor_decompress_fir_adapt(predictor_error, outputsamples ...)

                 * little strange..

                 */

            }

        }

    } else {

        /* not compressed, easy case */

        for (i = 0; i < outputsamples; i++) {

            for (ch = 0; ch < channels; ch++) {

                alac->outputsamples_buffer[ch][i] = get_sbits_long(&alac->gb,

                                                                   alac->setinfo_sample_size);

            }

        }

        alac->extra_bits = 0;

        interlacing_shift = 0;

        interlacing_leftweight = 0;

    }

    if (get_bits(&alac->gb, 3) != 7)

        av_log(avctx, AV_LOG_ERROR, "Error : Wrong End Of Frame\n");



    if (channels == 2 && interlacing_leftweight) {

        decorrelate_stereo(alac->outputsamples_buffer, outputsamples,

                           interlacing_shift, interlacing_leftweight);

    }



    if (alac->extra_bits) {

        append_extra_bits(alac->outputsamples_buffer, alac->extra_bits_buffer,

                          alac->extra_bits, alac->numchannels, outputsamples);

    }



    switch(alac->setinfo_sample_size) {

    case 16:

        if (channels == 2) {

            interleave_stereo_16(alac->outputsamples_buffer,

                                 (int16_t *)alac->frame.data[0], outputsamples);

        } else {

            int16_t *outbuffer = (int16_t *)alac->frame.data[0];

            for (i = 0; i < outputsamples; i++) {

                outbuffer[i] = alac->outputsamples_buffer[0][i];

            }

        }

        break;

    case 24:

        if (channels == 2) {

            interleave_stereo_24(alac->outputsamples_buffer,

                                 (int32_t *)alac->frame.data[0], outputsamples);

        } else {

            int32_t *outbuffer = (int32_t *)alac->frame.data[0];

            for (i = 0; i < outputsamples; i++)

                outbuffer[i] = alac->outputsamples_buffer[0][i] << 8;

        }

        break;

    }



    if (input_buffer_size * 8 - get_bits_count(&alac->gb) > 8)

        av_log(avctx, AV_LOG_ERROR, "Error : %d bits left\n", input_buffer_size * 8 - get_bits_count(&alac->gb));



    *got_frame_ptr   = 1;

    *(AVFrame *)data = alac->frame;



    return input_buffer_size;

}
