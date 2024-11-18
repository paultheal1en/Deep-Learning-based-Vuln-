static void mv88w8618_register_devices(void)

{

#ifdef HAS_AUDIO

    sysbus_register_withprop(&mv88w8618_audio_info);

#endif

}
