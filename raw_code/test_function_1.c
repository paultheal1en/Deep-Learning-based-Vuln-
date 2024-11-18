static void test_function(const TestStruct test_sample)

{

    int ret, i;

    void **output_data  = NULL;

    AVAudioFifo *afifo  = av_audio_fifo_alloc(test_sample.format, test_sample.nb_ch,

                                            test_sample.nb_samples_pch);

    if (!afifo) {

        ERROR("ERROR: av_audio_fifo_alloc returned NULL!");

    }

    ret = write_samples_to_audio_fifo(afifo, test_sample, test_sample.nb_samples_pch, 0);

    if (ret < 0){

        ERROR("ERROR: av_audio_fifo_write failed!");

    }

    printf("written: %d\n", ret);



    ret = write_samples_to_audio_fifo(afifo, test_sample, test_sample.nb_samples_pch, 0);

    if (ret < 0){

        ERROR("ERROR: av_audio_fifo_write failed!");

    }

    printf("written: %d\n", ret);

    printf("remaining samples in audio_fifo: %d\n\n", av_audio_fifo_size(afifo));



    ret = read_samples_from_audio_fifo(afifo, &output_data, test_sample.nb_samples_pch);

    if (ret < 0){

        ERROR("ERROR: av_audio_fifo_read failed!");

    }

    printf("read: %d\n", ret);

    print_audio_bytes(&test_sample, output_data, ret);

    printf("remaining samples in audio_fifo: %d\n\n", av_audio_fifo_size(afifo));



    /* test av_audio_fifo_peek */

    ret = av_audio_fifo_peek(afifo, output_data, afifo->nb_samples);

    if (ret < 0){

        ERROR("ERROR: av_audio_fifo_peek failed!");

    }

    printf("peek:\n");

    print_audio_bytes(&test_sample, output_data, ret);

    printf("\n");



    /* test av_audio_fifo_peek_at */

    printf("peek_at:\n");

    for (i = 0; i < afifo->nb_samples; ++i){

        ret = av_audio_fifo_peek_at(afifo, output_data, 1, i);

        if (ret < 0){

            ERROR("ERROR: av_audio_fifo_peek_at failed!");

        }

        printf("%d:\n", i);

        print_audio_bytes(&test_sample, output_data, ret);

    }

    printf("\n");



    /* test av_audio_fifo_drain */

    ret = av_audio_fifo_drain(afifo, afifo->nb_samples);

    if (ret < 0){

        ERROR("ERROR: av_audio_fifo_drain failed!");

    }

    if (afifo->nb_samples){

        ERROR("drain failed to flush all samples in audio_fifo!");

    }



    /* deallocate */

    for (i = 0; i < afifo->nb_buffers; ++i){

        av_freep(&output_data[i]);

    }

    av_freep(&output_data);

    av_audio_fifo_free(afifo);

}
