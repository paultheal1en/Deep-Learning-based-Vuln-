static void megasas_mmio_write(void *opaque, target_phys_addr_t addr,

                               uint64_t val, unsigned size)

{

    MegasasState *s = opaque;

    uint64_t frame_addr;

    uint32_t frame_count;

    int i;



    trace_megasas_mmio_writel(addr, val);

    switch (addr) {

    case MFI_IDB:

        if (val & MFI_FWINIT_ABORT) {

            /* Abort all pending cmds */

            for (i = 0; i < s->fw_cmds; i++) {

                megasas_abort_command(&s->frames[i]);

            }

        }

        if (val & MFI_FWINIT_READY) {

            /* move to FW READY */

            megasas_soft_reset(s);

        }

        if (val & MFI_FWINIT_MFIMODE) {

            /* discard MFIs */

        }

        break;

    case MFI_OMSK:

        s->intr_mask = val;

        if (!megasas_intr_enabled(s) && !msix_enabled(&s->dev)) {

            trace_megasas_irq_lower();

            qemu_irq_lower(s->dev.irq[0]);

        }

        if (megasas_intr_enabled(s)) {

            trace_megasas_intr_enabled();

        } else {

            trace_megasas_intr_disabled();

        }

        break;

    case MFI_ODCR0:

        s->doorbell = 0;

        if (s->producer_pa && megasas_intr_enabled(s)) {

            /* Update reply queue pointer */

            trace_megasas_qf_update(s->reply_queue_head, s->busy);

            stl_le_phys(s->producer_pa, s->reply_queue_head);

            if (!msix_enabled(&s->dev)) {

                trace_megasas_irq_lower();

                qemu_irq_lower(s->dev.irq[0]);

            }

        }

        break;

    case MFI_IQPH:

        /* Received high 32 bits of a 64 bit MFI frame address */

        s->frame_hi = val;

        break;

    case MFI_IQPL:

        /* Received low 32 bits of a 64 bit MFI frame address */

    case MFI_IQP:

        /* Received 32 bit MFI frame address */

        frame_addr = (val & ~0x1F);

        /* Add possible 64 bit offset */

        frame_addr |= ((uint64_t)s->frame_hi << 32);

        s->frame_hi = 0;

        frame_count = (val >> 1) & 0xF;

        megasas_handle_frame(s, frame_addr, frame_count);

        break;

    default:

        trace_megasas_mmio_invalid_writel(addr, val);

        break;

    }

}
