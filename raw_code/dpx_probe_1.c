static int dpx_probe(AVProbeData *p)

{

    const uint8_t *b = p->buf;



    if (AV_RN32(b) == AV_RN32("SDPX") || AV_RN32(b) == AV_RN32("XPDS"))

        return AVPROBE_SCORE_EXTENSION + 1;

    return 0;

}
