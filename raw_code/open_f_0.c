open_f(int argc, char **argv)

{

	int flags = 0;

	int readonly = 0;

	int growable = 0;

	int c;



	while ((c = getopt(argc, argv, "snrg")) != EOF) {

		switch (c) {

		case 's':

			flags |= BDRV_O_SNAPSHOT;

			break;

		case 'n':

			flags |= BDRV_O_NOCACHE;

			break;

		case 'r':

			readonly = 1;

			break;

		case 'g':

			growable = 1;

			break;

		default:

			return command_usage(&open_cmd);

		}

	}



	if (!readonly) {

            flags |= BDRV_O_RDWR;

        }



	if (optind != argc - 1)

		return command_usage(&open_cmd);



	return openfile(argv[optind], flags, growable);

}
