static int asf_probe(AVProbeData *pd)

{

    /* check file header */

    if (pd->buf_size <= 32)

        return 0;



    if (!memcmp(pd->buf, &asf_header, sizeof(GUID)))

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
