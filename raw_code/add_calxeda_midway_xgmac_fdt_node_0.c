static int add_calxeda_midway_xgmac_fdt_node(SysBusDevice *sbdev, void *opaque)

{

    PlatformBusFDTData *data = opaque;

    PlatformBusDevice *pbus = data->pbus;

    void *fdt = data->fdt;

    const char *parent_node = data->pbus_node_name;

    int compat_str_len, i, ret = -1;

    char *nodename;

    uint32_t *irq_attr, *reg_attr;

    uint64_t mmio_base, irq_number;

    VFIOPlatformDevice *vdev = VFIO_PLATFORM_DEVICE(sbdev);

    VFIODevice *vbasedev = &vdev->vbasedev;



    mmio_base = platform_bus_get_mmio_addr(pbus, sbdev, 0);

    nodename = g_strdup_printf("%s/%s@%" PRIx64, parent_node,

                               vbasedev->name, mmio_base);

    qemu_fdt_add_subnode(fdt, nodename);



    compat_str_len = strlen(vdev->compat) + 1;

    qemu_fdt_setprop(fdt, nodename, "compatible",

                          vdev->compat, compat_str_len);



    qemu_fdt_setprop(fdt, nodename, "dma-coherent", "", 0);



    reg_attr = g_new(uint32_t, vbasedev->num_regions * 2);

    for (i = 0; i < vbasedev->num_regions; i++) {

        mmio_base = platform_bus_get_mmio_addr(pbus, sbdev, i);

        reg_attr[2 * i] = cpu_to_be32(mmio_base);

        reg_attr[2 * i + 1] = cpu_to_be32(

                                memory_region_size(&vdev->regions[i]->mem));

    }

    ret = qemu_fdt_setprop(fdt, nodename, "reg", reg_attr,

                           vbasedev->num_regions * 2 * sizeof(uint32_t));

    if (ret) {

        error_report("could not set reg property of node %s", nodename);

        goto fail_reg;

    }



    irq_attr = g_new(uint32_t, vbasedev->num_irqs * 3);

    for (i = 0; i < vbasedev->num_irqs; i++) {

        irq_number = platform_bus_get_irqn(pbus, sbdev , i)

                         + data->irq_start;

        irq_attr[3 * i] = cpu_to_be32(GIC_FDT_IRQ_TYPE_SPI);

        irq_attr[3 * i + 1] = cpu_to_be32(irq_number);

        irq_attr[3 * i + 2] = cpu_to_be32(GIC_FDT_IRQ_FLAGS_LEVEL_HI);

    }

    ret = qemu_fdt_setprop(fdt, nodename, "interrupts",

                     irq_attr, vbasedev->num_irqs * 3 * sizeof(uint32_t));

    if (ret) {

        error_report("could not set interrupts property of node %s",

                     nodename);

    }

    g_free(irq_attr);

fail_reg:

    g_free(reg_attr);

    g_free(nodename);

    return ret;

}
