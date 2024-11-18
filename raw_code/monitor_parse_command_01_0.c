static const mon_cmd_t *monitor_parse_command(Monitor *mon,

                                              const char *cmdline,

                                              QDict *qdict)

{

    const char *p, *typestr;

    int c;

    const mon_cmd_t *cmd;

    char cmdname[256];

    char buf[1024];

    char *key;



#ifdef DEBUG

    monitor_printf(mon, "command='%s'\n", cmdline);

#endif



    /* extract the command name */

    p = get_command_name(cmdline, cmdname, sizeof(cmdname));

    if (!p)

        return NULL;



    /* find the command */

    for(cmd = mon_cmds; cmd->name != NULL; cmd++) {

        if (compare_cmd(cmdname, cmd->name))

            break;

    }



    if (cmd->name == NULL) {

        monitor_printf(mon, "unknown command: '%s'\n", cmdname);

        return NULL;

    }



    /* parse the parameters */

    typestr = cmd->args_type;

    for(;;) {

        typestr = key_get_info(typestr, &key);

        if (!typestr)

            break;

        c = *typestr;

        typestr++;

        switch(c) {

        case 'F':

        case 'B':

        case 's':

            {

                int ret;



                while (qemu_isspace(*p))

                    p++;

                if (*typestr == '?') {

                    typestr++;

                    if (*p == '\0') {

                        /* no optional string: NULL argument */

                        break;

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

                qdict_put(qdict, key, qstring_from_str(buf));

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

                qdict_put(qdict, "count", qint_from_int(count));

                qdict_put(qdict, "format", qint_from_int(format));

                qdict_put(qdict, "size", qint_from_int(size));

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

                        if (*p == '\0') {

                            typestr++;

                            break;

                        }

                    } else {

                        if (*p == '.') {

                            p++;

                            while (qemu_isspace(*p))

                                p++;

                        } else {

                            typestr++;

                            break;

                        }

                    }

                    typestr++;

                }

                if (get_expr(mon, &val, &p))

                    goto fail;

                /* Check if 'i' is greater than 32-bit */

                if ((c == 'i') && ((val >> 32) & 0xffffffff)) {

                    monitor_printf(mon, "\'%s\' has failed: ", cmdname);

                    monitor_printf(mon, "integer is for 32-bit values\n");

                    goto fail;

                }

                qdict_put(qdict, key, qint_from_int(val));

            }

            break;

        case '-':

            {

                const char *tmp = p;

                int has_option, skip_key = 0;

                /* option */



                c = *typestr++;

                if (c == '\0')

                    goto bad_type;

                while (qemu_isspace(*p))

                    p++;

                has_option = 0;

                if (*p == '-') {

                    p++;

                    if(c != *p) {

                        if(!is_valid_option(p, typestr)) {

                  

                            monitor_printf(mon, "%s: unsupported option -%c\n",

                                           cmdname, *p);

                            goto fail;

                        } else {

                            skip_key = 1;

                        }

                    }

                    if(skip_key) {

                        p = tmp;

                    } else {

                        p++;

                        has_option = 1;

                    }

                }

                qdict_put(qdict, key, qint_from_int(has_option));

            }

            break;

        default:

        bad_type:

            monitor_printf(mon, "%s: unknown type '%c'\n", cmdname, c);

            goto fail;

        }

        qemu_free(key);

        key = NULL;

    }

    /* check that all arguments were parsed */

    while (qemu_isspace(*p))

        p++;

    if (*p != '\0') {

        monitor_printf(mon, "%s: extraneous characters at the end of line\n",

                       cmdname);

        goto fail;

    }



    return cmd;



fail:

    qemu_free(key);

    return NULL;

}
