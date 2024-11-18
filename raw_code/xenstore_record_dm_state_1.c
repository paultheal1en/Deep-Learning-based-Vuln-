static void xenstore_record_dm_state(struct xs_handle *xs, const char *state)

{

    char path[50];



    if (xs == NULL) {

        fprintf(stderr, "xenstore connection not initialized\n");

        exit(1);

    }



    snprintf(path, sizeof (path), "/local/domain/0/device-model/%u/state", xen_domid);

    if (!xs_write(xs, XBT_NULL, path, state, strlen(state))) {

        fprintf(stderr, "error recording dm state\n");

        exit(1);

    }

}
