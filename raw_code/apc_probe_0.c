static int apc_probe(AVProbeData *p)

{

    if (p->buf_size < 8)

        return 0;



    if (!strncmp(p->buf, "CRYO_APC", 8))

        return AVPROBE_SCORE_MAX;



    return 0;

}
