void prepare_play(void)

{

    AVOutputFormat *ofmt;

    ofmt = guess_format("audio_device", NULL, NULL);

    if (!ofmt) {

        fprintf(stderr, "Could not find audio device\n");

        exit(1);

    }

    

    opt_output_file(audio_device);

}
