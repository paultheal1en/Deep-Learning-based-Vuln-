void ich9_lpc_pm_init(PCIDevice *lpc_pci)

{

    ICH9LPCState *lpc = ICH9_LPC_DEVICE(lpc_pci);



    ich9_pm_init(lpc_pci, &lpc->pm, qemu_allocate_irq(ich9_set_sci, lpc, 0));

    ich9_lpc_reset(&lpc->d.qdev);

}
