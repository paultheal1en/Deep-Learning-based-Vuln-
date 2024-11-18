static void default_drive(int enable, int snapshot, BlockInterfaceType type,

                          int index, const char *optstr)

{

    QemuOpts *opts;



    if (!enable || drive_get_by_index(type, index)) {

        return;

    }



    opts = drive_add(type, index, NULL, optstr);

    if (snapshot) {

        drive_enable_snapshot(opts, NULL);

    }

    if (!drive_new(opts, type)) {

        exit(1);

    }

}
