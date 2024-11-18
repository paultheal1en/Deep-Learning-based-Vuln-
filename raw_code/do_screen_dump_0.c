static void do_screen_dump(int argc, const char **argv)

{

    if (argc != 2) {

        help_cmd(argv[0]);

        return;

    }

    vga_screen_dump(argv[1]);

}
