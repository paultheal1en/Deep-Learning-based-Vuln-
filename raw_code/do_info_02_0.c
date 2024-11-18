static void do_info(int argc, const char **argv)

{

    term_cmd_t *cmd;

    const char *item;



    if (argc < 2)

        goto help;

    item = argv[1];

    for(cmd = info_cmds; cmd->name != NULL; cmd++) {

        if (compare_cmd(argv[1], cmd->name)) 

            goto found;

    }

 help:

    help_cmd(argv[0]);

    return;

 found:

    cmd->handler(argc, argv);

}
