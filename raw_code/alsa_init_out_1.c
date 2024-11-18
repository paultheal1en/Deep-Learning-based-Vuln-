static int alsa_init_out (HWVoiceOut *hw, struct audsettings *as)

{

    ALSAVoiceOut *alsa = (ALSAVoiceOut *) hw;

    struct alsa_params_req req;

    struct alsa_params_obt obt;

    snd_pcm_t *handle;

    struct audsettings obt_as;



    req.fmt = aud_to_alsafmt (as->fmt, as->endianness);

    req.freq = as->freq;

    req.nchannels = as->nchannels;

    req.period_size = conf.period_size_out;

    req.buffer_size = conf.buffer_size_out;

    req.size_in_usec = conf.size_in_usec_out;

    req.override_mask =

        (conf.period_size_out_overridden ? 1 : 0) |

        (conf.buffer_size_out_overridden ? 2 : 0);



    if (alsa_open (0, &req, &obt, &handle)) {

        return -1;

    }



    obt_as.freq = obt.freq;

    obt_as.nchannels = obt.nchannels;

    obt_as.fmt = obt.fmt;

    obt_as.endianness = obt.endianness;



    audio_pcm_init_info (&hw->info, &obt_as);

    hw->samples = obt.samples;



    alsa->pcm_buf = audio_calloc (AUDIO_FUNC, obt.samples, 1 << hw->info.shift);

    if (!alsa->pcm_buf) {

        dolog ("Could not allocate DAC buffer (%d samples, each %d bytes)\n",

               hw->samples, 1 << hw->info.shift);

        alsa_anal_close1 (&handle);

        return -1;

    }



    alsa->handle = handle;

    return 0;

}
