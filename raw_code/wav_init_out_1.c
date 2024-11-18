static int wav_init_out (HWVoiceOut *hw, struct audsettings *as)

{

    WAVVoiceOut *wav = (WAVVoiceOut *) hw;

    int bits16 = 0, stereo = 0;

    uint8_t hdr[] = {

        0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56,

        0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,

        0x02, 0x00, 0x44, 0xac, 0x00, 0x00, 0x10, 0xb1, 0x02, 0x00, 0x04,

        0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00

    };

    struct audsettings wav_as = conf.settings;



    (void) as;



    stereo = wav_as.nchannels == 2;

    switch (wav_as.fmt) {

    case AUD_FMT_S8:

    case AUD_FMT_U8:

        bits16 = 0;

        break;



    case AUD_FMT_S16:

    case AUD_FMT_U16:

        bits16 = 1;

        break;



    case AUD_FMT_S32:

    case AUD_FMT_U32:

        dolog ("WAVE files can not handle 32bit formats\n");

        return -1;

    }



    hdr[34] = bits16 ? 0x10 : 0x08;



    wav_as.endianness = 0;

    audio_pcm_init_info (&hw->info, &wav_as);



    hw->samples = 1024;

    wav->pcm_buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

    if (!wav->pcm_buf) {

        dolog ("Could not allocate buffer (%d bytes)\n",

               hw->samples << hw->info.shift);

        return -1;

    }



    le_store (hdr + 22, hw->info.nchannels, 2);

    le_store (hdr + 24, hw->info.freq, 4);

    le_store (hdr + 28, hw->info.freq << (bits16 + stereo), 4);

    le_store (hdr + 32, 1 << (bits16 + stereo), 2);



    wav->f = fopen (conf.wav_path, "wb");

    if (!wav->f) {

        dolog ("Failed to open wave file `%s'\nReason: %s\n",

               conf.wav_path, strerror (errno));

        g_free (wav->pcm_buf);

        wav->pcm_buf = NULL;

        return -1;

    }



    if (fwrite (hdr, sizeof (hdr), 1, wav->f) != 1) {

        dolog ("wav_init_out: failed to write header\nReason: %s\n",

               strerror(errno));

        return -1;

    }

    return 0;

}
