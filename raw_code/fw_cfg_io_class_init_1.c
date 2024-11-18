static void fw_cfg_io_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = fw_cfg_io_realize;

    dc->props = fw_cfg_io_properties;






}