static int acpi_pcihp_get_bsel(PCIBus *bus)

{

    QObject *o = object_property_get_qobject(OBJECT(bus),

                                             ACPI_PCIHP_PROP_BSEL, NULL);

    int64_t bsel = -1;

    if (o) {

        bsel = qint_get_int(qobject_to_qint(o));

    }

    if (bsel < 0) {

        return -1;

    }

    return bsel;

}
