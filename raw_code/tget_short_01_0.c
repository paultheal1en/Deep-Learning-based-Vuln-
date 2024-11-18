static unsigned tget_short(GetByteContext *gb, int le)

{

    unsigned v = le ? bytestream2_get_le16u(gb) : bytestream2_get_be16u(gb);

    return v;

}
