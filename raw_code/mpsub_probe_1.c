static int mpsub_probe(AVProbeData *p)

{

    const char *ptr     = p->buf;

    const char *ptr_end = p->buf + p->buf_size;



    while (ptr < ptr_end) {

        if (!memcmp(ptr, "FORMAT=TIME", 11))

            return AVPROBE_SCORE_EXTENSION;

        if (!memcmp(ptr, "FORMAT=", 7))

            return AVPROBE_SCORE_EXTENSION / 3;

        ptr += strcspn(ptr, "\n") + 1;

    }

    return 0;

}
