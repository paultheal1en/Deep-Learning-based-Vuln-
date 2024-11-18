static void allocate_buffers(ALACContext *alac)

{

    int chan;

    for (chan = 0; chan < alac->numchannels; chan++) {

        alac->predicterror_buffer[chan] =

            av_malloc(alac->setinfo_max_samples_per_frame * 4);



        alac->outputsamples_buffer[chan] =

            av_malloc(alac->setinfo_max_samples_per_frame * 4);



        alac->wasted_bits_buffer[chan] = av_malloc(alac->setinfo_max_samples_per_frame * 4);

    }

}
