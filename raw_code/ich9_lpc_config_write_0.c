static void ich9_lpc_config_write(PCIDevice *d,

                                  uint32_t addr, uint32_t val, int len)

{

    ICH9LPCState *lpc = ICH9_LPC_DEVICE(d);

    uint32_t rcba_old = pci_get_long(d->config + ICH9_LPC_RCBA);



    pci_default_write_config(d, addr, val, len);

    if (ranges_overlap(addr, len, ICH9_LPC_PMBASE, 4) ||

        ranges_overlap(addr, len, ICH9_LPC_ACPI_CTRL, 1)) {

        ich9_lpc_pmbase_sci_update(lpc);

    }

    if (ranges_overlap(addr, len, ICH9_LPC_RCBA, 4)) {

        ich9_lpc_rcba_update(lpc, rcba_old);

    }

    if (ranges_overlap(addr, len, ICH9_LPC_PIRQA_ROUT, 4)) {

        pci_bus_fire_intx_routing_notifier(lpc->d.bus);

    }

    if (ranges_overlap(addr, len, ICH9_LPC_PIRQE_ROUT, 4)) {

        pci_bus_fire_intx_routing_notifier(lpc->d.bus);

    }

    if (ranges_overlap(addr, len, ICH9_LPC_GEN_PMCON_1, 8)) {

        ich9_lpc_pmcon_update(lpc);

    }

}
