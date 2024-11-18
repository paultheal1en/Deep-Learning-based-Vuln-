static int msf_probe(AVProbeData *p)
{
    if (memcmp(p->buf, "MSF", 3))
        return 0;
    if (AV_RB32(p->buf+8) <= 0)
        return 0;
    if (AV_RB32(p->buf+16) <= 0)
        return 0;
    return AVPROBE_SCORE_MAX / 3 * 2;
}