static int piix3_pre_save(void *opaque)

{

    int i;

    PIIX3State *piix3 = opaque;



    for (i = 0; i < ARRAY_SIZE(piix3->pci_irq_levels_vmstate); i++) {

        piix3->pci_irq_levels_vmstate[i] =

            pci_bus_get_irq_level(piix3->dev.bus, i);

    }



    return 0;

}
