static void scsi_device_destroy(SCSIDevice *s)

{

    SCSIDeviceClass *sc = SCSI_DEVICE_GET_CLASS(s);

    if (sc->destroy) {

        sc->destroy(s);

    }

}
