void scsi_bus_legacy_handle_cmdline(SCSIBus *bus, Error **errp)

{

    Location loc;

    DriveInfo *dinfo;

    int unit;

    Error *err = NULL;



    loc_push_none(&loc);

    for (unit = 0; unit <= bus->info->max_target; unit++) {

        dinfo = drive_get(IF_SCSI, bus->busnr, unit);

        if (dinfo == NULL) {

            continue;

        }

        qemu_opts_loc_restore(dinfo->opts);

        scsi_bus_legacy_add_drive(bus, blk_bs(blk_by_legacy_dinfo(dinfo)),

                                  unit, false, -1, NULL, &err);

        if (err != NULL) {

            error_report("%s", error_get_pretty(err));

            error_propagate(errp, err);

            break;

        }

    }

    loc_pop(&loc);

}
