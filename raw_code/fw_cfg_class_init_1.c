static void fw_cfg_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = fw_cfg_realize;

    dc->no_user = 1;

    dc->reset = fw_cfg_reset;

    dc->vmsd = &vmstate_fw_cfg;

    dc->props = fw_cfg_properties;

}
