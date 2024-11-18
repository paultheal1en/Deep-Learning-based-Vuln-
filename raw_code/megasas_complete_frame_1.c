static void megasas_complete_frame(MegasasState *s, uint64_t context)

{

    PCIDevice *pci_dev = PCI_DEVICE(s);

    int tail, queue_offset;



    /* Decrement busy count */

    s->busy--;



    if (s->reply_queue_pa) {

        /*

         * Put command on the reply queue.

         * Context is opaque, but emulation is running in

         * little endian. So convert it.

         */

        tail = s->reply_queue_head;

        if (megasas_use_queue64(s)) {

            queue_offset = tail * sizeof(uint64_t);

            stq_le_phys(&address_space_memory,

                        s->reply_queue_pa + queue_offset, context);

        } else {

            queue_offset = tail * sizeof(uint32_t);

            stl_le_phys(&address_space_memory,

                        s->reply_queue_pa + queue_offset, context);

        }

        s->reply_queue_head = megasas_next_index(s, tail, s->fw_cmds);

        s->reply_queue_tail = ldl_le_phys(&address_space_memory,

                                          s->consumer_pa);

        trace_megasas_qf_complete(context, s->reply_queue_head,

                                  s->reply_queue_tail, s->busy, s->doorbell);

    }



    if (megasas_intr_enabled(s)) {

        /* Notify HBA */

        s->doorbell++;

        if (s->doorbell == 1) {

            if (msix_enabled(pci_dev)) {

                trace_megasas_msix_raise(0);

                msix_notify(pci_dev, 0);

            } else if (msi_enabled(pci_dev)) {

                trace_megasas_msi_raise(0);

                msi_notify(pci_dev, 0);

            } else {

                trace_megasas_irq_raise();

                pci_irq_assert(pci_dev);

            }

        }

    } else {

        trace_megasas_qf_complete_noirq(context);

    }

}
