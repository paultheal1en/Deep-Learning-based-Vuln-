static void audio_print_settings (audsettings_t *as)

{

    dolog ("frequency=%d nchannels=%d fmt=", as->freq, as->nchannels);



    switch (as->fmt) {

    case AUD_FMT_S8:

        AUD_log (NULL, "S8");

        break;

    case AUD_FMT_U8:

        AUD_log (NULL, "U8");

        break;

    case AUD_FMT_S16:

        AUD_log (NULL, "S16");

        break;

    case AUD_FMT_U16:

        AUD_log (NULL, "U16");

        break;

    case AUD_FMT_S32:

        AUD_log (NULL, "S32");

        break;

    case AUD_FMT_U32:

        AUD_log (NULL, "U32");

        break;

    default:

        AUD_log (NULL, "invalid(%d)", as->fmt);

        break;

    }



    AUD_log (NULL, " endianness=");

    switch (as->endianness) {

    case 0:

        AUD_log (NULL, "little");

        break;

    case 1:

        AUD_log (NULL, "big");

        break;

    default:

        AUD_log (NULL, "invalid");

        break;

    }

    AUD_log (NULL, "\n");

}
