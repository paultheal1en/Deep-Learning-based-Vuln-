static void test_commands(void)

{

    char *response;

    int i;



    for (i = 0; hmp_cmds[i] != NULL; i++) {

        if (verbose) {

            fprintf(stderr, "\t%s\n", hmp_cmds[i]);

        }

        response = hmp(hmp_cmds[i]);

        g_free(response);

    }



}
