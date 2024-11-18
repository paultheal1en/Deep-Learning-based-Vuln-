static int sdl_init_out (HWVoiceOut *hw, audsettings_t *as)

{

    SDLVoiceOut *sdl = (SDLVoiceOut *) hw;

    SDLAudioState *s = &glob_sdl;

    SDL_AudioSpec req, obt;

    int shift;

    int endianess;

    int err;

    audfmt_e effective_fmt;

    audsettings_t obt_as;



    shift <<= as->nchannels == 2;



    req.freq = as->freq;

    req.format = aud_to_sdlfmt (as->fmt, &shift);

    req.channels = as->nchannels;

    req.samples = conf.nb_samples;

    req.callback = sdl_callback;

    req.userdata = sdl;



    if (sdl_open (&req, &obt)) {

        return -1;

    }



    err = sdl_to_audfmt (obt.format, &effective_fmt, &endianess);

    if (err) {

        sdl_close (s);

        return -1;

    }



    obt_as.freq = obt.freq;

    obt_as.nchannels = obt.channels;

    obt_as.fmt = effective_fmt;

    obt_as.endianness = endianess;



    audio_pcm_init_info (&hw->info, &obt_as);

    hw->samples = obt.samples;



    s->initialized = 1;

    s->exit = 0;

    SDL_PauseAudio (0);

    return 0;

}
