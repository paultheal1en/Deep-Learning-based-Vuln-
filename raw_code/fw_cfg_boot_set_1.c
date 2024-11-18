static void fw_cfg_boot_set(void *opaque, const char *boot_device,

                            Error **errp)

{

    fw_cfg_add_i16(opaque, FW_CFG_BOOT_DEVICE, boot_device[0]);

}
