static void fw_cfg_mem_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = fw_cfg_mem_realize;

    dc->props = fw_cfg_mem_properties;






}