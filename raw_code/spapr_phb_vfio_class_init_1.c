static void spapr_phb_vfio_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    sPAPRPHBClass *spc = SPAPR_PCI_HOST_BRIDGE_CLASS(klass);



    dc->props = spapr_phb_vfio_properties;

    spc->finish_realize = spapr_phb_vfio_finish_realize;

}
