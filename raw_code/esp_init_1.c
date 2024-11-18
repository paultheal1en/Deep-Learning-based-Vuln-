void esp_init(target_phys_addr_t espaddr, int it_shift,

              espdma_memory_read_write dma_memory_read,

              espdma_memory_read_write dma_memory_write,

              void *dma_opaque, qemu_irq irq, qemu_irq *reset)

{

    DeviceState *dev;

    SysBusDevice *s;

    ESPState *esp;



    dev = qdev_create(NULL, "esp");

    esp = DO_UPCAST(ESPState, busdev.qdev, dev);

    esp->dma_memory_read = dma_memory_read;

    esp->dma_memory_write = dma_memory_write;

    esp->dma_opaque = dma_opaque;

    esp->it_shift = it_shift;

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    sysbus_connect_irq(s, 0, irq);

    sysbus_mmio_map(s, 0, espaddr);

    *reset = qdev_get_gpio_in(dev, 0);

}
