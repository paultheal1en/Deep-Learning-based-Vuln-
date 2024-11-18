void sysbus_dev_print(Monitor *mon, DeviceState *dev, int indent)

{

    SysBusDevice *s = sysbus_from_qdev(dev);

    int i;



    for (i = 0; i < s->num_mmio; i++) {

        monitor_printf(mon, "%*smmio " TARGET_FMT_plx "/" TARGET_FMT_plx "\n",

                       indent, "", s->mmio[i].addr, s->mmio[i].size);

    }

}
