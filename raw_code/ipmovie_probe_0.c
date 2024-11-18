static int ipmovie_probe(AVProbeData *p)

{

    if (p->buf_size < IPMOVIE_SIGNATURE_SIZE)

        return 0;

    if (strncmp(p->buf, IPMOVIE_SIGNATURE, IPMOVIE_SIGNATURE_SIZE) != 0)

        return 0;



    return AVPROBE_SCORE_MAX;

}
