static int swf_probe(AVProbeData *p)

{

    if(p->buf_size < 15)

        return 0;



    /* check file header */

    if (   AV_RB24(p->buf) != AV_RB24("CWS")

        && AV_RB24(p->buf) != AV_RB24("FWS"))

        return 0;



    if (p->buf[3] >= 20)

        return AVPROBE_SCORE_MAX / 4;



    return AVPROBE_SCORE_MAX;

}
