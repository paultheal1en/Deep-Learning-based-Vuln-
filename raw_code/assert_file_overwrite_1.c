static void assert_file_overwrite(const char *filename)

{

    if (file_overwrite && no_file_overwrite) {

        fprintf(stderr, "Error, both -y and -n supplied. Exiting.\n");

        exit_program(1);

    }



    if (!file_overwrite &&

        (strchr(filename, ':') == NULL || filename[1] == ':' ||

         av_strstart(filename, "file:", NULL))) {

        if (avio_check(filename, 0) == 0) {

            if (stdin_interaction && !no_file_overwrite) {

                fprintf(stderr,"File '%s' already exists. Overwrite ? [y/N] ", filename);

                fflush(stderr);

                term_exit();

                signal(SIGINT, SIG_DFL);

                if (!read_yesno()) {

                    av_log(NULL, AV_LOG_FATAL, "Not overwriting - exiting\n");

                    exit_program(1);

                }

                term_init();

            }

            else {

                av_log(NULL, AV_LOG_FATAL, "File '%s' already exists. Exiting.\n", filename);

                exit_program(1);

            }

        }

    }

}
