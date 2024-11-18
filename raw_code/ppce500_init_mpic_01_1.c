static qemu_irq *ppce500_init_mpic(PPCE500Params *params, MemoryRegion *ccsr,

                                   qemu_irq **irqs)

{

    QemuOptsList *list;

    qemu_irq *mpic;

    DeviceState *dev = NULL;

    SysBusDevice *s;

    int i;



    mpic = g_new(qemu_irq, 256);



    if (kvm_enabled()) {

        bool irqchip_allowed = true, irqchip_required = false;



        list = qemu_find_opts("machine");

        if (!QTAILQ_EMPTY(&list->head)) {

            irqchip_allowed = qemu_opt_get_bool(QTAILQ_FIRST(&list->head),

                                                "kernel_irqchip", true);

            irqchip_required = qemu_opt_get_bool(QTAILQ_FIRST(&list->head),

                                                 "kernel_irqchip", false);

        }



        if (irqchip_allowed) {

            dev = ppce500_init_mpic_kvm(params, irqs);

        }



        if (irqchip_required && !dev) {

            fprintf(stderr, "%s: irqchip requested but unavailable\n",

                    __func__);

            abort();

        }

    }



    if (!dev) {

        dev = ppce500_init_mpic_qemu(params, irqs);

    }



    for (i = 0; i < 256; i++) {

        mpic[i] = qdev_get_gpio_in(dev, i);

    }



    s = SYS_BUS_DEVICE(dev);

    memory_region_add_subregion(ccsr, MPC8544_MPIC_REGS_OFFSET,

                                s->mmio[0].memory);



    return mpic;

}
