static int audio_validate_settings (audsettings_t *as)

{

    int invalid;



    invalid = as->nchannels != 1 && as->nchannels != 2;

    invalid |= as->endianness != 0 && as->endianness != 1;



    switch (as->fmt) {

    case AUD_FMT_S8:

    case AUD_FMT_U8:

    case AUD_FMT_S16:

    case AUD_FMT_U16:

    case AUD_FMT_S32:

    case AUD_FMT_U32:

        break;

    default:

        invalid = 1;

        break;

    }



    invalid |= as->freq <= 0;

    return invalid ? -1 : 0;

}
