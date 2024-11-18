static void opt_format(const char *arg)

{

    /* compatibility stuff for pgmyuv */

    if (!strcmp(arg, "pgmyuv")) {

        opt_image_format(arg);

        arg = "image";

    }



    file_iformat = av_find_input_format(arg);

    file_oformat = guess_format(arg, NULL, NULL);

    if (!file_iformat && !file_oformat) {

        fprintf(stderr, "Unknown input or output format: %s\n", arg);

        exit(1);

    }

}
