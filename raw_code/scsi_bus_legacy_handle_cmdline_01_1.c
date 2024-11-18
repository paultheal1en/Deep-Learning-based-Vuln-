void scsi_bus_legacy_handle_cmdline(SCSIBus *bus)

{

    DriveInfo *dinfo;

    int unit;



    for (unit = 0; unit < MAX_SCSI_DEVS; unit++) {

        dinfo = drive_get(IF_SCSI, bus->busnr, unit);

        if (dinfo == NULL) {

            continue;

        }

        scsi_bus_legacy_add_drive(bus, dinfo, unit);

    }

}
