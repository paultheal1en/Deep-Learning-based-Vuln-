static void do_savevm(int argc, const char **argv)

{

    if (argc != 2) {

        help_cmd(argv[0]);

        return;

    }

    if (qemu_savevm(argv[1]) < 0)

        term_printf("I/O error when saving VM to '%s'\n", argv[1]);

}
