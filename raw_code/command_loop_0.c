command_loop(void)

{

	int		c, i, j = 0, done = 0;

	char		*input;

	char		**v;

	const cmdinfo_t	*ct;



	for (i = 0; !done && i < ncmdline; i++) {

		input = strdup(cmdline[i]);

		if (!input) {

			fprintf(stderr,

				_("cannot strdup command '%s': %s\n"),

				cmdline[i], strerror(errno));

			exit(1);

		}

		v = breakline(input, &c);

		if (c) {

			ct = find_command(v[0]);

			if (ct) {

				if (ct->flags & CMD_FLAG_GLOBAL)

					done = command(ct, c, v);

				else {

					j = 0;

					while (!done && (j = args_command(j)))

						done = command(ct, c, v);

				}

			} else

				fprintf(stderr, _("command \"%s\" not found\n"),

					v[0]);

		}

		doneline(input, v);

	}

	if (cmdline) {

		free(cmdline);

		return;

	}

	while (!done) {

		if ((input = fetchline()) == NULL)

			break;

		v = breakline(input, &c);

		if (c) {

			ct = find_command(v[0]);

			if (ct)

				done = command(ct, c, v);

			else

				fprintf(stderr, _("command \"%s\" not found\n"),

					v[0]);

		}

		doneline(input, v);

	}

}
