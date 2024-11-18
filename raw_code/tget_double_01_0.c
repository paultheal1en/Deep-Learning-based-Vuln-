static double tget_double(const uint8_t **p, int le)

{

    av_alias64 i = { .u64 = le ? AV_RL64(*p) : AV_RB64(*p)};

    *p += 8;

    return i.f64;

}
