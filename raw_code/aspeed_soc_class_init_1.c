static void aspeed_soc_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    AspeedSoCClass *sc = ASPEED_SOC_CLASS(oc);



    sc->info = (AspeedSoCInfo *) data;

    dc->realize = aspeed_soc_realize;



}