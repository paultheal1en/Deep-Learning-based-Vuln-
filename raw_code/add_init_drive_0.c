DriveInfo *add_init_drive(const char *optstr)

{

    DriveInfo *dinfo;

    QemuOpts *opts;



    opts = drive_def(optstr);

    if (!opts)

        return NULL;



    dinfo = drive_init(opts, current_machine->use_scsi);

    if (!dinfo) {

        qemu_opts_del(opts);

        return NULL;

    }



    return dinfo;

}
