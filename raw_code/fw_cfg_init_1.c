FWCfgState *fw_cfg_init(uint32_t ctl_port, uint32_t data_port,

                        hwaddr ctl_addr, hwaddr data_addr)

{

    DeviceState *dev;

    SysBusDevice *d;

    FWCfgState *s;



    dev = qdev_create(NULL, TYPE_FW_CFG);

    qdev_prop_set_uint32(dev, "ctl_iobase", ctl_port);

    qdev_prop_set_uint32(dev, "data_iobase", data_port);

    d = SYS_BUS_DEVICE(dev);



    s = FW_CFG(dev);



    assert(!object_resolve_path(FW_CFG_PATH, NULL));



    object_property_add_child(qdev_get_machine(), FW_CFG_NAME, OBJECT(s), NULL);



    qdev_init_nofail(dev);



    if (ctl_addr) {

        sysbus_mmio_map(d, 0, ctl_addr);

    }

    if (data_addr) {

        sysbus_mmio_map(d, 1, data_addr);

    }

    fw_cfg_add_bytes(s, FW_CFG_SIGNATURE, (char *)"QEMU", 4);

    fw_cfg_add_bytes(s, FW_CFG_UUID, qemu_uuid, 16);

    fw_cfg_add_i16(s, FW_CFG_NOGRAPHIC, (uint16_t)(display_type == DT_NOGRAPHIC));

    fw_cfg_add_i16(s, FW_CFG_NB_CPUS, (uint16_t)smp_cpus);

    fw_cfg_add_i16(s, FW_CFG_BOOT_MENU, (uint16_t)boot_menu);

    fw_cfg_bootsplash(s);

    fw_cfg_reboot(s);



    s->machine_ready.notify = fw_cfg_machine_ready;

    qemu_add_machine_init_done_notifier(&s->machine_ready);



    return s;

}
