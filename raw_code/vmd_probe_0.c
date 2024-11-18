static int vmd_probe(AVProbeData *p)

{

    if (p->buf_size < 2)

        return 0;



    /* check if the first 2 bytes of the file contain the appropriate size

     * of a VMD header chunk */

    if (AV_RL16(&p->buf[0]) != VMD_HEADER_SIZE - 2)

        return 0;



    /* only return half certainty since this check is a bit sketchy */

    return AVPROBE_SCORE_MAX / 2;

}
