static int line_out_init (HWVoiceOut *hw, struct audsettings *as)

{

    SpiceVoiceOut *out = container_of (hw, SpiceVoiceOut, hw);

    struct audsettings settings;



#if SPICE_INTERFACE_PLAYBACK_MAJOR > 1 || SPICE_INTERFACE_PLAYBACK_MINOR >= 3

    settings.freq       = spice_server_get_best_playback_rate(NULL);

#else

    settings.freq       = SPICE_INTERFACE_PLAYBACK_FREQ;

#endif

    settings.nchannels  = SPICE_INTERFACE_PLAYBACK_CHAN;

    settings.fmt        = AUD_FMT_S16;

    settings.endianness = AUDIO_HOST_ENDIANNESS;



    audio_pcm_init_info (&hw->info, &settings);

    hw->samples = LINE_OUT_SAMPLES;

    out->active = 0;



    out->sin.base.sif = &playback_sif.base;

    qemu_spice_add_interface (&out->sin.base);

#if SPICE_INTERFACE_PLAYBACK_MAJOR > 1 || SPICE_INTERFACE_PLAYBACK_MINOR >= 3

    spice_server_set_playback_rate(&out->sin, settings.freq);

#endif

    return 0;

}
