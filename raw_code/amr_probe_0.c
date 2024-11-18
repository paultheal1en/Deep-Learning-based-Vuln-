static int amr_probe(AVProbeData *p)

{

    //Only check for "#!AMR" which could be amr-wb, amr-nb.

    //This will also trigger multichannel files: "#!AMR_MC1.0\n" and

    //"#!AMR-WB_MC1.0\n" (not supported)



    if (p->buf_size < 5)

        return 0;

    if(memcmp(p->buf,AMR_header,5)==0)

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
