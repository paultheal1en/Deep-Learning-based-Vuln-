static int qpa_init_in (HWVoiceIn *hw, struct audsettings *as)

{

    int error;

    static pa_sample_spec ss;

    struct audsettings obt_as = *as;

    PAVoiceIn *pa = (PAVoiceIn *) hw;



    ss.format = audfmt_to_pa (as->fmt, as->endianness);

    ss.channels = as->nchannels;

    ss.rate = as->freq;



    obt_as.fmt = pa_to_audfmt (ss.format, &obt_as.endianness);



    pa->s = pa_simple_new (

        conf.server,

        "qemu",

        PA_STREAM_RECORD,

        conf.source,

        "pcm.capture",

        &ss,

        NULL,                   /* channel map */

        NULL,                   /* buffering attributes */

        &error

        );

    if (!pa->s) {

        qpa_logerr (error, "pa_simple_new for capture failed\n");

        goto fail1;

    }



    audio_pcm_init_info (&hw->info, &obt_as);

    hw->samples = conf.samples;

    pa->pcm_buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

    pa->wpos = hw->wpos;

    if (!pa->pcm_buf) {

        dolog ("Could not allocate buffer (%d bytes)\n",

               hw->samples << hw->info.shift);

        goto fail2;

    }



    if (audio_pt_init (&pa->pt, qpa_thread_in, hw, AUDIO_CAP, AUDIO_FUNC)) {

        goto fail3;

    }



    return 0;



 fail3:

    g_free (pa->pcm_buf);

    pa->pcm_buf = NULL;

 fail2:

    pa_simple_free (pa->s);

    pa->s = NULL;

 fail1:

    return -1;

}
