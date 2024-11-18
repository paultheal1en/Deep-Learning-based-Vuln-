static int adx_decode(ADXContext *c, int16_t *out, int offset,

                      const uint8_t *in, int ch)

{

    ADXChannelState *prev = &c->prev[ch];

    GetBitContext gb;

    int scale = AV_RB16(in);

    int i;

    int s0, s1, s2, d;



    /* check if this is an EOF packet */

    if (scale & 0x8000)

        return -1;



    init_get_bits(&gb, in + 2, (BLOCK_SIZE - 2) * 8);

    out += offset;

    s1 = prev->s1;

    s2 = prev->s2;

    for (i = 0; i < BLOCK_SAMPLES; i++) {

        d  = get_sbits(&gb, 4);

        s0 = ((d << COEFF_BITS) * scale + c->coeff[0] * s1 + c->coeff[1] * s2) >> COEFF_BITS;

        s2 = s1;

        s1 = av_clip_int16(s0);

        *out++ = s1;

    }

    prev->s1 = s1;

    prev->s2 = s2;



    return 0;

}
