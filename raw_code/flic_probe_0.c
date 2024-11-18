static int flic_probe(AVProbeData *p)

{

    int magic_number;



    if (p->buf_size < 6)

        return 0;



    magic_number = AV_RL16(&p->buf[4]);

    if ((magic_number != FLIC_FILE_MAGIC_1) &&

        (magic_number != FLIC_FILE_MAGIC_2) &&

        (magic_number != FLIC_FILE_MAGIC_3))

        return 0;



    return AVPROBE_SCORE_MAX;

}
