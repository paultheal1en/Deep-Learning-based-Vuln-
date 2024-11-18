void scsi_bus_legacy_handle_cmdline(SCSIBus *bus, bool deprecated)

{

    Location loc;

    DriveInfo *dinfo;

    int unit;



    loc_push_none(&loc);

    for (unit = 0; unit <= bus->info->max_target; unit++) {

        dinfo = drive_get(IF_SCSI, bus->busnr, unit);

        if (dinfo == NULL) {

            continue;

        }

        qemu_opts_loc_restore(dinfo->opts);

        if (deprecated) {

            /* Handling -drive not claimed by machine initialization */

            if (blk_get_attached_dev(blk_by_legacy_dinfo(dinfo))) {

                continue;       /* claimed */

            }

            if (!dinfo->is_default) {

                error_report("warning: bus=%d,unit=%d is deprecated with this"

                             " machine type",

                             bus->busnr, unit);

            }

        }

        scsi_bus_legacy_add_drive(bus, blk_by_legacy_dinfo(dinfo),

                                  unit, false, -1, NULL, &error_fatal);

    }

    loc_pop(&loc);

}
