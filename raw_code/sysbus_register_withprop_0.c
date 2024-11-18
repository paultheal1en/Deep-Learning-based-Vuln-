void sysbus_register_withprop(SysBusDeviceInfo *info)

{

    info->qdev.init = sysbus_device_init;

    info->qdev.bus_type = BUS_TYPE_SYSTEM;



    assert(info->qdev.size >= sizeof(SysBusDevice));

    qdev_register(&info->qdev);

}
