static int waveformat_to_audio_settings (WAVEFORMATEX *wfx, audsettings_t *as)

{

    if (wfx->wFormatTag != WAVE_FORMAT_PCM) {

        dolog ("Invalid wave format, tag is not PCM, but %d\n",

               wfx->wFormatTag);

        return -1;

    }



    if (!wfx->nSamplesPerSec) {

        dolog ("Invalid wave format, frequency is zero\n");

        return -1;

    }

    as->freq = wfx->nSamplesPerSec;



    switch (wfx->nChannels) {

    case 1:

        as->nchannels = 1;

        break;



    case 2:

        as->nchannels = 2;

        break;



    default:

        dolog (

            "Invalid wave format, number of channels is not 1 or 2, but %d\n",

            wfx->nChannels

            );

        return -1;

    }



    switch (wfx->wBitsPerSample) {

    case 8:

        as->fmt = AUD_FMT_U8;

        break;



    case 16:

        as->fmt = AUD_FMT_S16;

        break;



    case 32:

        as->fmt = AUD_FMT_S32;

        break;



    default:

        dolog ("Invalid wave format, bits per sample is not "

               "8, 16 or 32, but %d\n",

               wfx->wBitsPerSample);

        return -1;

    }



    return 0;

}
