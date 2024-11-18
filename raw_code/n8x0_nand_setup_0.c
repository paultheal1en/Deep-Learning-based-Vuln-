static void n8x0_nand_setup(struct n800_s *s)

{

    char *otp_region;

    DriveInfo *dinfo;



    s->nand = qdev_create(NULL, "onenand");

    qdev_prop_set_uint16(s->nand, "manufacturer_id", NAND_MFR_SAMSUNG);

    /* Either 0x40 or 0x48 are OK for the device ID */

    qdev_prop_set_uint16(s->nand, "device_id", 0x48);

    qdev_prop_set_uint16(s->nand, "version_id", 0);

    qdev_prop_set_int32(s->nand, "shift", 1);

    dinfo = drive_get(IF_MTD, 0, 0);

    if (dinfo) {

        qdev_prop_set_drive_nofail(s->nand, "drive",

                                   blk_bs(blk_by_legacy_dinfo(dinfo)));

    }

    qdev_init_nofail(s->nand);

    sysbus_connect_irq(SYS_BUS_DEVICE(s->nand), 0,

                       qdev_get_gpio_in(s->mpu->gpio, N8X0_ONENAND_GPIO));

    omap_gpmc_attach(s->mpu->gpmc, N8X0_ONENAND_CS,

                     sysbus_mmio_get_region(SYS_BUS_DEVICE(s->nand), 0));

    otp_region = onenand_raw_otp(s->nand);



    memcpy(otp_region + 0x000, n8x0_cal_wlan_mac, sizeof(n8x0_cal_wlan_mac));

    memcpy(otp_region + 0x800, n8x0_cal_bt_id, sizeof(n8x0_cal_bt_id));

    /* XXX: in theory should also update the OOB for both pages */

}
