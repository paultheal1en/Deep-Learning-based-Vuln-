static const MXFCodecUL *mxf_get_essence_container_ul(enum CodecID type)

{

    const MXFCodecUL *uls = ff_mxf_essence_container_uls;

    while (uls->id != CODEC_ID_NONE) {

        if (uls->id == type)

            break;

        uls++;

    }

    return uls;

}
