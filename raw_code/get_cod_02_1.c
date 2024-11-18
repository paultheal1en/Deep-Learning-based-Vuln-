static int get_cod(Jpeg2000DecoderContext *s, Jpeg2000CodingStyle *c,
                   uint8_t *properties)
{
    Jpeg2000CodingStyle tmp;
    int compno, ret;
    if (bytestream2_get_bytes_left(&s->g) < 5)
    tmp.csty = bytestream2_get_byteu(&s->g);
    // get progression order
    tmp.prog_order = bytestream2_get_byteu(&s->g);
    tmp.nlayers    = bytestream2_get_be16u(&s->g);
    tmp.mct        = bytestream2_get_byteu(&s->g); // multiple component transformation
    if ((ret = get_cox(s, &tmp)) < 0)
        return ret;
    for (compno = 0; compno < s->ncomponents; compno++)
        if (!(properties[compno] & HAD_COC))
            memcpy(c + compno, &tmp, sizeof(tmp));
    return 0;