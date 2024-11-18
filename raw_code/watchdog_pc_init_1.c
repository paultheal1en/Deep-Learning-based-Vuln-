void watchdog_pc_init(PCIBus *pci_bus)

{

    if (watchdog)

        watchdog->wdt_pc_init(pci_bus);

}
