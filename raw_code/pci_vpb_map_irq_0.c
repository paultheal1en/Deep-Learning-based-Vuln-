static int pci_vpb_map_irq(PCIDevice *d, int irq_num)

{

    PCIVPBState *s = container_of(d->bus, PCIVPBState, pci_bus);



    if (s->irq_mapping == PCI_VPB_IRQMAP_BROKEN) {

        /* Legacy broken IRQ mapping for compatibility with old and

         * buggy Linux guests

         */

        return irq_num;

    }



    /* Slot to IRQ mapping for RealView Platform Baseboard 926 backplane

     *      name    slot    IntA    IntB    IntC    IntD

     *      A       31      IRQ28   IRQ29   IRQ30   IRQ27

     *      B       30      IRQ27   IRQ28   IRQ29   IRQ30

     *      C       29      IRQ30   IRQ27   IRQ28   IRQ29

     * Slot C is for the host bridge; A and B the peripherals.

     * Our output irqs 0..3 correspond to the baseboard's 27..30.

     *

     * This mapping function takes account of an oddity in the PB926

     * board wiring, where the FPGA's P_nINTA input is connected to

     * the INTB connection on the board PCI edge connector, P_nINTB

     * is connected to INTC, and so on, so everything is one number

     * further round from where you might expect.

     */

    return pci_swizzle_map_irq_fn(d, irq_num + 2);

}
