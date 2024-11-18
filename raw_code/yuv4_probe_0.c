static int yuv4_probe(AVProbeData *pd)

{

    /* check file header */

    if (pd->buf_size <= sizeof(Y4M_MAGIC))

        return 0;

    if (strncmp(pd->buf, Y4M_MAGIC, sizeof(Y4M_MAGIC)-1)==0)

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
