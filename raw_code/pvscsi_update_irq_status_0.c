pvscsi_update_irq_status(PVSCSIState *s)

{

    PCIDevice *d = PCI_DEVICE(s);

    bool should_raise = s->reg_interrupt_enabled & s->reg_interrupt_status;



    trace_pvscsi_update_irq_level(should_raise, s->reg_interrupt_enabled,

                                  s->reg_interrupt_status);



    if (s->msi_used && msi_enabled(d)) {

        if (should_raise) {

            trace_pvscsi_update_irq_msi();

            msi_notify(d, PVSCSI_VECTOR_COMPLETION);

        }

        return;

    }



    pci_set_irq(d, !!should_raise);

}
