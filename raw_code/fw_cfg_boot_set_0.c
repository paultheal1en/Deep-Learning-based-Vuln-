static int fw_cfg_boot_set(void *opaque, const char *boot_device)

{

    fw_cfg_add_i16(opaque, FW_CFG_BOOT_DEVICE, boot_device[0]);

    return 0;

}
