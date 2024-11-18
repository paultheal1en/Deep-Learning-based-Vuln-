static int waveformat_from_audio_settings (WAVEFORMATEX *wfx, audsettings_t *as)

{

    memset (wfx, 0, sizeof (*wfx));



    wfx->wFormatTag = WAVE_FORMAT_PCM;

    wfx->nChannels = as->nchannels;

    wfx->nSamplesPerSec = as->freq;

    wfx->nAvgBytesPerSec = as->freq << (as->nchannels == 2);

    wfx->nBlockAlign = 1 << (as->nchannels == 2);

    wfx->cbSize = 0;



    switch (as->fmt) {

    case AUD_FMT_S8:

    case AUD_FMT_U8:

        wfx->wBitsPerSample = 8;

        break;



    case AUD_FMT_S16:

    case AUD_FMT_U16:

        wfx->wBitsPerSample = 16;

        wfx->nAvgBytesPerSec <<= 1;

        wfx->nBlockAlign <<= 1;

        break;



    case AUD_FMT_S32:

    case AUD_FMT_U32:

        wfx->wBitsPerSample = 32;

        wfx->nAvgBytesPerSec <<= 2;

        wfx->nBlockAlign <<= 2;

        break;



    default:

        dolog ("Internal logic error: Bad audio format %d\n", as->freq);

        return -1;

    }



    return 0;

}
