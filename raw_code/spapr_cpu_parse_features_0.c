void spapr_cpu_parse_features(sPAPRMachineState *spapr)

{

    /*

     * Backwards compatibility hack:

     *

     *   CPUs had a "compat=" property which didn't make sense for

     *   anything except pseries.  It was replaced by "max-cpu-compat"

     *   machine option.  This supports old command lines like

     *       -cpu POWER8,compat=power7

     *   By stripping the compat option and applying it to the machine

     *   before passing it on to the cpu level parser.

     */

    gchar **inpieces;

    gchar *newprops;

    int i, j;

    gchar *compat_str = NULL;



    inpieces = g_strsplit(MACHINE(spapr)->cpu_model, ",", 0);



    /* inpieces[0] is the actual model string */

    i = 1;

    j = 1;

    while (inpieces[i]) {

        if (g_str_has_prefix(inpieces[i], "compat=")) {

            /* in case of multiple compat= options */

            g_free(compat_str);

            compat_str = inpieces[i];

        } else {

            j++;

        }



        i++;

        /* Excise compat options from list */

        inpieces[j] = inpieces[i];

    }



    if (compat_str) {

        char *val = compat_str + strlen("compat=");



        object_property_set_str(OBJECT(spapr), val, "max-cpu-compat",

                                &error_fatal);



    }



    newprops = g_strjoinv(",", inpieces);

    cpu_parse_cpu_model(TYPE_POWERPC_CPU, newprops);

    g_free(newprops);

    g_strfreev(inpieces);

}
