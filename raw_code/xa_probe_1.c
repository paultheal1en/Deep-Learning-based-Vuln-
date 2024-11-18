static int xa_probe(AVProbeData *p)

{

    switch(AV_RL32(p->buf)) {

    case XA00_TAG:

    case XAI0_TAG:

    case XAJ0_TAG:

        return AVPROBE_SCORE_MAX;

    }

    return 0;

}
