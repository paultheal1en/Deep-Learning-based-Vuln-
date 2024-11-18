static PCIINTxRoute gpex_route_intx_pin_to_irq(void *opaque, int pin)

{

    PCIINTxRoute route;

    GPEXHost *s = opaque;



    route.mode = PCI_INTX_ENABLED;

    route.irq = s->irq_num[pin];



    return route;

}
