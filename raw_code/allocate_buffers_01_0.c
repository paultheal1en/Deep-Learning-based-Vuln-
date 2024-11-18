static int allocate_buffers(ALACContext *alac)

{

    int ch;

    for (ch = 0; ch < FFMIN(alac->channels, 2); ch++) {

        int buf_size = alac->max_samples_per_frame * sizeof(int32_t);



        FF_ALLOC_OR_GOTO(alac->avctx, alac->predict_error_buffer[ch],

                         buf_size, buf_alloc_fail);



        if (alac->sample_size == 16) {

            FF_ALLOC_OR_GOTO(alac->avctx, alac->output_samples_buffer[ch],

                             buf_size, buf_alloc_fail);

        }



        FF_ALLOC_OR_GOTO(alac->avctx, alac->extra_bits_buffer[ch],

                         buf_size, buf_alloc_fail);

    }

    return 0;

buf_alloc_fail:

    alac_decode_close(alac->avctx);

    return AVERROR(ENOMEM);

}
