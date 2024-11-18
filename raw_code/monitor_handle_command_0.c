static void monitor_handle_command(Monitor *mon, const char *cmdline)

{

    const char *p, *pstart, *typestr;

    char *q;

    int c, nb_args, len, i, has_arg;

    const mon_cmd_t *cmd;

    char cmdname[256];

    char buf[1024];

    void *str_allocated[MAX_ARGS];

    void *args[MAX_ARGS];

    void (*handler_0)(Monitor *mon);

    void (*handler_1)(Monitor *mon, void *arg0);

    void (*handler_2)(Monitor *mon, void *arg0, void *arg1);

    void (*handler_3)(Monitor *mon, void *arg0, void *arg1, void *arg2);

    void (*handler_4)(Monitor *mon, void *arg0, void *arg1, void *arg2,

                      void *arg3);

    void (*handler_5)(Monitor *mon, void *arg0, void *arg1, void *arg2,

                      void *arg3, void *arg4);

    void (*handler_6)(Monitor *mon, void *arg0, void *arg1, void *arg2,

                      void *arg3, void *arg4, void *arg5);

    void (*handler_7)(Monitor *mon, void *arg0, void *arg1, void *arg2,

                      void *arg3, void *arg4, void *arg5, void *arg6);



#ifdef DEBUG

    monitor_printf(mon, "command='%s'\n", cmdline);

#endif



    /* extract the command name */

    p = cmdline;

    q = cmdname;

    while (qemu_isspace(*p))

        p++;

    if (*p == '\0')

        return;

    pstart = p;

    while (*p != '\0' && *p != '/' && !qemu_isspace(*p))

        p++;

    len = p - pstart;

    if (len > sizeof(cmdname) - 1)

        len = sizeof(cmdname) - 1;

    memcpy(cmdname, pstart, len);

    cmdname[len] = '\0';



    /* find the command */

    for(cmd = mon_cmds; cmd->name != NULL; cmd++) {

        if (compare_cmd(cmdname, cmd->name))

            goto found;

    }

    monitor_printf(mon, "unknown command: '%s'\n", cmdname);

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



                while (qemu_isspace(*p))

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

                        monitor_printf(mon, "%s: filename expected\n",

                                       cmdname);

                        break;

                    case 'B':

                        monitor_printf(mon, "%s: block device name expected\n",

                                       cmdname);

                        break;

                    default:

                        monitor_printf(mon, "%s: string expected\n", cmdname);

                        break;

                    }

                    goto fail;

                }

                str = qemu_malloc(strlen(buf) + 1);

                pstrcpy(str, sizeof(buf), buf);

                str_allocated[nb_args] = str;

            add_str:

                if (nb_args >= MAX_ARGS) {

                error_args:

                    monitor_printf(mon, "%s: too many arguments\n", cmdname);

                    goto fail;

                }

                args[nb_args++] = str;

            }

            break;

        case '/':

            {

                int count, format, size;



                while (qemu_isspace(*p))

                    p++;

                if (*p == '/') {

                    /* format found */

                    p++;

                    count = 1;

                    if (qemu_isdigit(*p)) {

                        count = 0;

                        while (qemu_isdigit(*p)) {

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

                    if (*p != '\0' && !qemu_isspace(*p)) {

                        monitor_printf(mon, "invalid char in format: '%c'\n",

                                       *p);

                        goto fail;

                    }

                    if (format < 0)

                        format = default_fmt_format;

                    if (format != 'i') {

                        /* for 'i', not specifying a size gives -1 as size */

                        if (size < 0)

                            size = default_fmt_size;

                        default_fmt_size = size;

                    }

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

                args[nb_args++] = (void*)(long)count;

                args[nb_args++] = (void*)(long)format;

                args[nb_args++] = (void*)(long)size;

            }

            break;

        case 'i':

        case 'l':

            {

                int64_t val;



                while (qemu_isspace(*p))

                    p++;

                if (*typestr == '?' || *typestr == '.') {

                    if (*typestr == '?') {

                        if (*p == '\0')

                            has_arg = 0;

                        else

                            has_arg = 1;

                    } else {

                        if (*p == '.') {

                            p++;

                            while (qemu_isspace(*p))

                                p++;

                            has_arg = 1;

                        } else {

                            has_arg = 0;

                        }

                    }

                    typestr++;

                    if (nb_args >= MAX_ARGS)

                        goto error_args;

                    args[nb_args++] = (void *)(long)has_arg;

                    if (!has_arg) {

                        if (nb_args >= MAX_ARGS)

                            goto error_args;

                        val = -1;

                        goto add_num;

                    }

                }

                if (get_expr(mon, &val, &p))

                    goto fail;

            add_num:

                if (c == 'i') {

                    if (nb_args >= MAX_ARGS)

                        goto error_args;

                    args[nb_args++] = (void *)(long)val;

                } else {

                    if ((nb_args + 1) >= MAX_ARGS)

                        goto error_args;

#if TARGET_PHYS_ADDR_BITS > 32

                    args[nb_args++] = (void *)(long)((val >> 32) & 0xffffffff);

#else

                    args[nb_args++] = (void *)0;

#endif

                    args[nb_args++] = (void *)(long)(val & 0xffffffff);

                }

            }

            break;

        case '-':

            {

                int has_option;

                /* option */



                c = *typestr++;

                if (c == '\0')

                    goto bad_type;

                while (qemu_isspace(*p))

                    p++;

                has_option = 0;

                if (*p == '-') {

                    p++;

                    if (*p != c) {

                        monitor_printf(mon, "%s: unsupported option -%c\n",

                                       cmdname, *p);

                        goto fail;

                    }

                    p++;

                    has_option = 1;

                }

                if (nb_args >= MAX_ARGS)

                    goto error_args;

                args[nb_args++] = (void *)(long)has_option;

            }

            break;

        default:

        bad_type:

            monitor_printf(mon, "%s: unknown type '%c'\n", cmdname, c);

            goto fail;

        }

    }

    /* check that all arguments were parsed */

    while (qemu_isspace(*p))

        p++;

    if (*p != '\0') {

        monitor_printf(mon, "%s: extraneous characters at the end of line\n",

                       cmdname);

        goto fail;

    }



    switch(nb_args) {

    case 0:

        handler_0 = cmd->handler;

        handler_0(mon);

        break;

    case 1:

        handler_1 = cmd->handler;

        handler_1(mon, args[0]);

        break;

    case 2:

        handler_2 = cmd->handler;

        handler_2(mon, args[0], args[1]);

        break;

    case 3:

        handler_3 = cmd->handler;

        handler_3(mon, args[0], args[1], args[2]);

        break;

    case 4:

        handler_4 = cmd->handler;

        handler_4(mon, args[0], args[1], args[2], args[3]);

        break;

    case 5:

        handler_5 = cmd->handler;

        handler_5(mon, args[0], args[1], args[2], args[3], args[4]);

        break;

    case 6:

        handler_6 = cmd->handler;

        handler_6(mon, args[0], args[1], args[2], args[3], args[4], args[5]);

        break;

    case 7:

        handler_7 = cmd->handler;

        handler_7(mon, args[0], args[1], args[2], args[3], args[4], args[5],

                  args[6]);

        break;

    default:

        monitor_printf(mon, "unsupported number of arguments: %d\n", nb_args);

        goto fail;

    }

 fail:

    for(i = 0; i < MAX_ARGS; i++)

        qemu_free(str_allocated[i]);

    return;

}
