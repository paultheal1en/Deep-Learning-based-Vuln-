static void test_info_commands(void)

{

    char *resp, *info, *info_buf, *endp;



    info_buf = info = hmp("help info");



    while (*info) {

        /* Extract the info command, ignore parameters and description */

        g_assert(strncmp(info, "info ", 5) == 0);

        endp = strchr(&info[5], ' ');

        g_assert(endp != NULL);

        *endp = '\0';

        /* Now run the info command */

        if (verbose) {

            fprintf(stderr, "\t%s\n", info);

        }

        resp = hmp(info);

        g_free(resp);

        /* And move forward to the next line */

        info = strchr(endp + 1, '\n');

        if (!info) {

            break;

        }

        info += 1;

    }



    g_free(info_buf);

}
