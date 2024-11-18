static int dirac_probe(AVProbeData *p)

{

    if (AV_RL32(p->buf) == MKTAG('B', 'B', 'C', 'D'))

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
