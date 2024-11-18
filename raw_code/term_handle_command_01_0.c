static void term_handle_command(const char *cmdline)

{

    const char *p, *pstart, *typestr;

    char *q;

    int c, nb_args, len, i, has_arg;

    term_cmd_t *cmd;

    char cmdname[256];

    char buf[1024];

    void *str_allocated[MAX_ARGS];

    void *args[MAX_ARGS];



#ifdef DEBUG

    term_printf("command='%s'\n", cmdline);

#endif

    

    /* extract the command name */

    p = cmdline;

    q = cmdname;

    while (isspace(*p))

        p++;

    if (*p == '\0')

        return;

    pstart = p;

    while (*p != '\0' && *p != '/' && !isspace(*p))

        p++;

    len = p - pstart;

    if (len > sizeof(cmdname) - 1)

        len = sizeof(cmdname) - 1;

    memcpy(cmdname, pstart, len);

    cmdname[len] = '\0';

    

    /* find the command */

    for(cmd = term_cmds; cmd->name != NULL; cmd++) {

        if (compare_cmd(cmdname, cmd->name)) 

            goto found;

    }

    term_printf("unknown command: '%s'\n", cmdname);

    return;

 found:



    for(i = 0; i < MAX_ARGS; i++)

        str_allocated[i] = NULL;

    

    /* parse the parameters */

    typestr = cmd->args_type;

    nb_args = 0;

    for(;;) {

        c = *typestr;

        if (c == '\0')

            break;

        typestr++;

        switch(c) {

        case 'F':

        case 'B':

        case 's':

            {

                int ret;

                char *str;

                

                while (isspace(*p)) 

                    p++;

                if (*typestr == '?') {

                    typestr++;

                    if (*p == '\0') {

                        /* no optional string: NULL argument */

                        str = NULL;

                        goto add_str;

                    }

                }

                ret = get_str(buf, sizeof(buf), &p);

                if (ret < 0) {

                    switch(c) {

                    case 'F':

                        term_printf("%s: filename expected\n", cmdname);

                        break;

                    case 'B':

                        term_printf("%s: block device name expected\n", cmdname);

                        break;

                    default:

                        term_printf("%s: string expected\n", cmdname);

                        break;

                    }

                    goto fail;

                }

                str = qemu_malloc(strlen(buf) + 1);

                strcpy(str, buf);

                str_allocated[nb_args] = str;

            add_str:

                if (nb_args >= MAX_ARGS) {

                error_args:

                    term_printf("%s: too many arguments\n", cmdname);

                    goto fail;

                }

                args[nb_args++] = str;

            }

            break;

        case '/':

            {

                int count, format, size;

                

                while (isspace(*p))

                    p++;

                if (*p == '/') {

                    /* format found */

                    p++;

                    count = 1;

                    if (isdigit(*p)) {

                        count = 0;

                        while (isdigit(*p)) {

                            count = count * 10 + (*p - '0');

                            p++;

                        }

                    }

                    size = -1;

                    format = -1;

                    for(;;) {

                        switch(*p) {

                        case 'o':

                        case 'd':

                        case 'u':

                        case 'x':

                        case 'i':

                        case 'c':

                            format = *p++;

                            break;

                        case 'b':

                            size = 1;

                            p++;

                            break;

                        case 'h':

                            size = 2;

                            p++;

                            break;

                        case 'w':

                            size = 4;

                            p++;

                            break;

                        case 'g':

                        case 'L':

                            size = 8;

                            p++;

                            break;

                        default:

                            goto next;

                        }

                    }

                next:

                    if (*p != '\0' && !isspace(*p)) {

                        term_printf("invalid char in format: '%c'\n", *p);

                        goto fail;

                    }

                    if (format < 0)

                        format = default_fmt_format;

                    if (format != 'i') {

                        /* for 'i', not specifying a size gives -1 as size */

                        if (size < 0)

                            size = default_fmt_size;

                    }

                    default_fmt_size = size;

                    default_fmt_format = format;

                } else {

                    count = 1;

                    format = default_fmt_format;

                    if (format != 'i') {

                        size = default_fmt_size;

                    } else {

                        size = -1;

                    }

                }

                if (nb_args + 3 > MAX_ARGS)

                    goto error_args;

                args[nb_args++] = (void*)count;

                args[nb_args++] = (void*)format;

                args[nb_args++] = (void*)size;

            }

            break;

        case 'i':

            {

                int val;

                while (isspace(*p)) 

                    p++;

                if (*typestr == '?' || *typestr == '.') {

                    typestr++;

                    if (*typestr == '?') {

                        if (*p == '\0')

                            has_arg = 0;

                        else

                            has_arg = 1;

                    } else {

                        if (*p == '.') {

                            p++;

                            while (isspace(*p)) 

                                p++;

                            has_arg = 1;

                        } else {

                            has_arg = 0;

                        }

                    }

                    if (nb_args >= MAX_ARGS)

                        goto error_args;

                    args[nb_args++] = (void *)has_arg;

                    if (!has_arg) {

                        if (nb_args >= MAX_ARGS)

                            goto error_args;

                        val = -1;

                        goto add_num;

                    }

                }

                if (get_expr(&val, &p))

                    goto fail;

            add_num:

                if (nb_args >= MAX_ARGS)

                    goto error_args;

                args[nb_args++] = (void *)val;

            }

            break;

        case '-':

            {

                int has_option;

                /* option */

                

                c = *typestr++;

                if (c == '\0')

                    goto bad_type;

                while (isspace(*p)) 

                    p++;

                has_option = 0;

                if (*p == '-') {

                    p++;

                    if (*p != c) {

                        term_printf("%s: unsupported option -%c\n", 

                                    cmdname, *p);

                        goto fail;

                    }

                    p++;

                    has_option = 1;

                }

                if (nb_args >= MAX_ARGS)

                    goto error_args;

                args[nb_args++] = (void *)has_option;

            }

            break;

        default:

        bad_type:

            term_printf("%s: unknown type '%c'\n", cmdname, c);

            goto fail;

        }

    }

    /* check that all arguments were parsed */

    while (isspace(*p))

        p++;

    if (*p != '\0') {

        term_printf("%s: extraneous characters at the end of line\n", 

                    cmdname);

        goto fail;

    }



    switch(nb_args) {

    case 0:

        cmd->handler();

        break;

    case 1:

        cmd->handler(args[0]);

        break;

    case 2:

        cmd->handler(args[0], args[1]);

        break;

    case 3:

        cmd->handler(args[0], args[1], args[2]);

        break;

    case 4:

        cmd->handler(args[0], args[1], args[2], args[3]);

        break;

    case 5:

        cmd->handler(args[0], args[1], args[2], args[3], args[4]);

        break;

    case 6:

        cmd->handler(args[0], args[1], args[2], args[3], args[4], args[5]);

        break;

    default:

        term_printf("unsupported number of arguments: %d\n", nb_args);

        goto fail;

    }

 fail:

    for(i = 0; i < MAX_ARGS; i++)

        qemu_free(str_allocated[i]);

    return;

}
