static const MXFCodecUL *mxf_get_codec_ul(const MXFCodecUL *uls, UID *uid)

{

    while (uls->id != CODEC_ID_NONE) {

        if(mxf_match_uid(uls->uid, *uid, 16))

            break;

        uls++;

    }

    return uls;

}
