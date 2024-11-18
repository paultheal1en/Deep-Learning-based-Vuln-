REORDER_OUT_50(int8, int8_t)

REORDER_OUT_51(int8, int8_t)

REORDER_OUT_71(int8, int8_t)

REORDER_OUT_50(int16, int16_t)

REORDER_OUT_51(int16, int16_t)

REORDER_OUT_71(int16, int16_t)

REORDER_OUT_50(int32, int32_t)

REORDER_OUT_51(int32, int32_t)

REORDER_OUT_71(int32, int32_t)

REORDER_OUT_50(f32, float)

REORDER_OUT_51(f32, float)

REORDER_OUT_71(f32, float)



#define FORMAT_I8  0

#define FORMAT_I16 1

#define FORMAT_I32 2

#define FORMAT_F32 3



#define PICK_REORDER(layout)\

switch(format) {\

    case FORMAT_I8:  s->reorder_func = alsa_reorder_int8_out_ ##layout;  break;\

    case FORMAT_I16: s->reorder_func = alsa_reorder_int16_out_ ##layout; break;\

    case FORMAT_I32: s->reorder_func = alsa_reorder_int32_out_ ##layout; break;\

    case FORMAT_F32: s->reorder_func = alsa_reorder_f32_out_ ##layout;   break;\

}



static av_cold int find_reorder_func(AlsaData *s, int codec_id, int64_t layout, int out)

{

    int format;



    /* reordering input is not currently supported */

    if (!out)

        return AVERROR(ENOSYS);



    /* reordering is not needed for QUAD or 2_2 layout */

    if (layout == AV_CH_LAYOUT_QUAD || layout == AV_CH_LAYOUT_2_2)

        return 0;



    switch (codec_id) {

    case CODEC_ID_PCM_S8:

    case CODEC_ID_PCM_U8:

    case CODEC_ID_PCM_ALAW:

    case CODEC_ID_PCM_MULAW: format = FORMAT_I8;  break;

    case CODEC_ID_PCM_S16LE:

    case CODEC_ID_PCM_S16BE:

    case CODEC_ID_PCM_U16LE:

    case CODEC_ID_PCM_U16BE: format = FORMAT_I16; break;

    case CODEC_ID_PCM_S32LE:

    case CODEC_ID_PCM_S32BE:

    case CODEC_ID_PCM_U32LE:

    case CODEC_ID_PCM_U32BE: format = FORMAT_I32; break;

    case CODEC_ID_PCM_F32LE:

    case CODEC_ID_PCM_F32BE: format = FORMAT_F32; break;

    default:                 return AVERROR(ENOSYS);

    }



    if      (layout == AV_CH_LAYOUT_5POINT0_BACK || layout == AV_CH_LAYOUT_5POINT0)

        PICK_REORDER(50)

    else if (layout == AV_CH_LAYOUT_5POINT1_BACK || layout == AV_CH_LAYOUT_5POINT1)

        PICK_REORDER(51)

    else if (layout == AV_CH_LAYOUT_7POINT1)

        PICK_REORDER(71)



    return s->reorder_func ? 0 : AVERROR(ENOSYS);

}
