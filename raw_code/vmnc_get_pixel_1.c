static av_always_inline int vmnc_get_pixel(const uint8_t *buf, int bpp, int be)

{

    switch (bpp * 2 + be) {

    case 2:

    case 3:

        return *buf;

    case 4:

        return AV_RL16(buf);

    case 5:

        return AV_RB16(buf);

    case 8:

        return AV_RL32(buf);

    case 9:

        return AV_RB32(buf);

    default:

        return 0;

    }

}
