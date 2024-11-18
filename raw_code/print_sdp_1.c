static void print_sdp(void)

{

    char sdp[16384];

    int i;

    AVFormatContext **avc = av_malloc(sizeof(*avc) * nb_output_files);



    if (!avc)

        exit(1);

    for (i = 0; i < nb_output_files; i++)

        avc[i] = output_files[i]->ctx;



    av_sdp_create(avc, nb_output_files, sdp, sizeof(sdp));

    printf("SDP:\n%s\n", sdp);

    fflush(stdout);

    av_freep(&avc);

}
