static bool iasl_installed(void)

{

    gchar *out = NULL, *out_err = NULL;

    bool ret;



    /* pass 'out' and 'out_err' in order to be redirected */

    ret = g_spawn_command_line_sync("iasl", &out, &out_err, NULL, NULL);



    if (out_err) {

        ret = ret && (out_err[0] == '\0');

        g_free(out_err);

    }



    if (out) {

        g_free(out);

    }



    return ret;

}
