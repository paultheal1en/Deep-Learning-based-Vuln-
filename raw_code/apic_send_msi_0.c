static void apic_send_msi(target_phys_addr_t addr, uint32_t data)

{

    uint8_t dest = (addr & MSI_ADDR_DEST_ID_MASK) >> MSI_ADDR_DEST_ID_SHIFT;

    uint8_t vector = (data & MSI_DATA_VECTOR_MASK) >> MSI_DATA_VECTOR_SHIFT;

    uint8_t dest_mode = (addr >> MSI_ADDR_DEST_MODE_SHIFT) & 0x1;

    uint8_t trigger_mode = (data >> MSI_DATA_TRIGGER_SHIFT) & 0x1;

    uint8_t delivery = (data >> MSI_DATA_DELIVERY_MODE_SHIFT) & 0x7;

    /* XXX: Ignore redirection hint. */

    apic_deliver_irq(dest, dest_mode, delivery, vector, trigger_mode);

}
