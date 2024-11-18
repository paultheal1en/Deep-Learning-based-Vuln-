void command_loop(void)

{

    int c, i, j = 0, done = 0, fetchable = 0, prompted = 0;

    char *input;

    char **v;

    const cmdinfo_t *ct;



    for (i = 0; !done && i < ncmdline; i++) {

        input = strdup(cmdline[i]);

        if (!input) {

            fprintf(stderr, _("cannot strdup command '%s': %s\n"),

                    cmdline[i], strerror(errno));

            exit(1);

        }

        v = breakline(input, &c);

        if (c) {

            ct = find_command(v[0]);

            if (ct) {

                if (ct->flags & CMD_FLAG_GLOBAL) {

                    done = command(ct, c, v);

                } else {

                    j = 0;

                    while (!done && (j = args_command(j))) {

                        done = command(ct, c, v);

                    }

                }

            } else {

                fprintf(stderr, _("command \"%s\" not found\n"), v[0]);

            }

	}

        doneline(input, v);

    }

    if (cmdline) {

        free(cmdline);

        return;

    }



    while (!done) {

        if (!prompted) {

            printf("%s", get_prompt());

            fflush(stdout);

            qemu_aio_set_fd_handler(STDIN_FILENO, prep_fetchline, NULL, NULL,

                                    NULL, &fetchable);

            prompted = 1;

        }



        qemu_aio_wait();



        if (!fetchable) {

            continue;

        }

        input = fetchline();

        if (input == NULL) {

            break;

        }

        v = breakline(input, &c);

        if (c) {

            ct = find_command(v[0]);

            if (ct) {

                done = command(ct, c, v);

            } else {

                fprintf(stderr, _("command \"%s\" not found\n"), v[0]);

            }

        }

        doneline(input, v);



        prompted = 0;

        fetchable = 0;

    }

    qemu_aio_set_fd_handler(STDIN_FILENO, NULL, NULL, NULL, NULL, NULL);

}
