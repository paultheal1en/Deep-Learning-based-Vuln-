bool sysbus_has_irq(SysBusDevice *dev, int n)
{
    char *prop = g_strdup_printf("%s[%d]", SYSBUS_DEVICE_GPIO_IRQ, n);
    ObjectProperty *r;
    r = object_property_find(OBJECT(dev), prop, NULL);
    return (r != NULL);
}