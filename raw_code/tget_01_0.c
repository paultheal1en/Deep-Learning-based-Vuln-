static unsigned tget(GetByteContext *gb, int type, int le)

{

    switch (type) {

    case TIFF_BYTE : return bytestream2_get_byteu(gb);

    case TIFF_SHORT: return tget_short(gb, le);

    case TIFF_LONG : return tget_long(gb, le);

    default        : return UINT_MAX;

    }

}
