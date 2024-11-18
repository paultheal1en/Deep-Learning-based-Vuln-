static int srt_probe(AVProbeData *p)

{

    const unsigned char *ptr = p->buf;

    int i, v, num = 0;



    if (AV_RB24(ptr) == 0xEFBBBF)

        ptr += 3;  /* skip UTF-8 BOM */



    while (*ptr == '\r' || *ptr == '\n')

        ptr++;

    for (i=0; i<2; i++) {

        if ((num == i || num + 1 == i)

            && sscanf(ptr, "%*d:%*2d:%*2d%*1[,.]%*3d --> %*d:%*2d:%*2d%*1[,.]%3d", &v) == 1)

            return AVPROBE_SCORE_MAX;

        num = atoi(ptr);

        ptr += strcspn(ptr, "\n") + 1;

    }

    return 0;

}
