static void do_log(int argc, const char **argv)

{

    int mask;

    

    if (argc != 2)

        goto help;

    if (!strcmp(argv[1], "none")) {

        mask = 0;

    } else {

        mask = cpu_str_to_log_mask(argv[1]);

        if (!mask) {

        help:

            help_cmd(argv[0]);

            return;

        }

    }

    cpu_set_log(mask);

}
