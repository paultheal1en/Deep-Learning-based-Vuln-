static void fsl_imx6_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->realize = fsl_imx6_realize;



    dc->desc = "i.MX6 SOC";

}
