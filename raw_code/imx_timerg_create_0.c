void imx_timerg_create(const target_phys_addr_t addr,

                              qemu_irq irq,

                              DeviceState *ccm)

{

    IMXTimerGState *pp;

    DeviceState *dev;



    dev = sysbus_create_simple("imx_timerg", addr, irq);

    pp = container_of(dev, IMXTimerGState, busdev.qdev);

    pp->ccm = ccm;

}
