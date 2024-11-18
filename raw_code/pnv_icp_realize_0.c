static void pnv_icp_realize(DeviceState *dev, Error **errp)

{

    PnvICPState *icp = PNV_ICP(dev);



    memory_region_init_io(&icp->mmio, OBJECT(dev), &pnv_icp_ops,

                          icp, "icp-thread", 0x1000);

}
