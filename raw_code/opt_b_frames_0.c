static void opt_b_frames(const char *arg)

{

    b_frames = atoi(arg);

    if (b_frames > FF_MAX_B_FRAMES) {

        fprintf(stderr, "\nCannot have more than %d B frames, increase FF_MAX_B_FRAMES.\n", FF_MAX_B_FRAMES);

        exit(1);

    } else if (b_frames < 1) {

        fprintf(stderr, "\nNumber of B frames must be higher than 0\n");

        exit(1);

    }

}
