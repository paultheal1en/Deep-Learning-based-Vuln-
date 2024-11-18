int audio_available(void)

{

#ifdef HAS_AUDIO

    return 1;

#else

    return 0;

#endif

}
