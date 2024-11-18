static void mixer_reset (AC97LinkState *s)

{

    uint8_t active[LAST_INDEX];



    dolog ("mixer_reset\n");

    memset (s->mixer_data, 0, sizeof (s->mixer_data));

    memset (active, 0, sizeof (active));

    mixer_store (s, AC97_Reset                   , 0x0000); /* 6940 */

    mixer_store (s, AC97_Master_Volume_Mono_Mute , 0x8000);

    mixer_store (s, AC97_PC_BEEP_Volume_Mute     , 0x0000);



    mixer_store (s, AC97_Phone_Volume_Mute       , 0x8008);

    mixer_store (s, AC97_Mic_Volume_Mute         , 0x8008);

    mixer_store (s, AC97_CD_Volume_Mute          , 0x8808);

    mixer_store (s, AC97_Aux_Volume_Mute         , 0x8808);

    mixer_store (s, AC97_Record_Gain_Mic_Mute    , 0x8000);

    mixer_store (s, AC97_General_Purpose         , 0x0000);

    mixer_store (s, AC97_3D_Control              , 0x0000);

    mixer_store (s, AC97_Powerdown_Ctrl_Stat     , 0x000f);



    /*

     * Sigmatel 9700 (STAC9700)

     */

    mixer_store (s, AC97_Vendor_ID1              , 0x8384);

    mixer_store (s, AC97_Vendor_ID2              , 0x7600); /* 7608 */



    mixer_store (s, AC97_Extended_Audio_ID       , 0x0809);

    mixer_store (s, AC97_Extended_Audio_Ctrl_Stat, 0x0009);

    mixer_store (s, AC97_PCM_Front_DAC_Rate      , 0xbb80);

    mixer_store (s, AC97_PCM_Surround_DAC_Rate   , 0xbb80);

    mixer_store (s, AC97_PCM_LFE_DAC_Rate        , 0xbb80);

    mixer_store (s, AC97_PCM_LR_ADC_Rate         , 0xbb80);

    mixer_store (s, AC97_MIC_ADC_Rate            , 0xbb80);



    record_select (s, 0);

    set_volume (s, AC97_Master_Volume_Mute, 0x8000);

    set_volume (s, AC97_PCM_Out_Volume_Mute, 0x8808);

    set_volume (s, AC97_Line_In_Volume_Mute, 0x8808);



    reset_voices (s, active);

}
