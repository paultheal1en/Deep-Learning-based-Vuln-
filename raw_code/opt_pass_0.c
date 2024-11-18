static void opt_pass(const char *pass_str)

{

    int pass;

    pass = atoi(pass_str);

    if (pass != 1 && pass != 2) {

        fprintf(stderr, "pass number can be only 1 or 2\n");

        ffmpeg_exit(1);

    }

    do_pass = pass;

}
