static void find_completion(const char *cmdline)

{

    const char *cmdname;

    char *args[MAX_ARGS];

    int nb_args, i, len;

    const char *ptype, *str;

    term_cmd_t *cmd;



    parse_cmdline(cmdline, &nb_args, args);

#ifdef DEBUG_COMPLETION

    for(i = 0; i < nb_args; i++) {

        term_printf("arg%d = '%s'\n", i, (char *)args[i]);

    }

#endif



    /* if the line ends with a space, it means we want to complete the

       next arg */

    len = strlen(cmdline);

    if (len > 0 && isspace(cmdline[len - 1])) {

        if (nb_args >= MAX_ARGS)

            return;

        args[nb_args++] = qemu_strdup("");

    }

    if (nb_args <= 1) {

        /* command completion */

        if (nb_args == 0)

            cmdname = "";

        else

            cmdname = args[0];

        completion_index = strlen(cmdname);

        for(cmd = term_cmds; cmd->name != NULL; cmd++) {

            cmd_completion(cmdname, cmd->name);

        }

    } else {

        /* find the command */

        for(cmd = term_cmds; cmd->name != NULL; cmd++) {

            if (compare_cmd(args[0], cmd->name))

                goto found;

        }

        return;

    found:

        ptype = cmd->args_type;

        for(i = 0; i < nb_args - 2; i++) {

            if (*ptype != '\0') {

                ptype++;

                while (*ptype == '?')

                    ptype++;

            }

        }

        str = args[nb_args - 1];

        switch(*ptype) {

        case 'F':

            /* file completion */

            completion_index = strlen(str);

            file_completion(str);

            break;

        case 'B':

            /* block device name completion */

            completion_index = strlen(str);

            bdrv_iterate(block_completion_it, (void *)str);

            break;

        default:

            break;

        }

    }

    for(i = 0; i < nb_args; i++)

        qemu_free(args[i]);

}
