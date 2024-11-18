static void opt_input_file(void *optctx, const char *arg)

{

    if (input_filename) {

        fprintf(stderr,

                "Argument '%s' provided as input filename, but '%s' was already specified.\n",

                arg, input_filename);

        exit(1);

    }

    if (!strcmp(arg, "-"))

        arg = "pipe:";

    input_filename = arg;

}
