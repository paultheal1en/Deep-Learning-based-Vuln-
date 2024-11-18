static unsigned tget_short(const uint8_t **p, int le)

{

    unsigned v = le ? AV_RL16(*p) : AV_RB16(*p);

    *p += 2;

    return v;

}
