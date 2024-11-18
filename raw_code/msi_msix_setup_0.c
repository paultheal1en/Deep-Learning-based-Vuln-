static int msi_msix_setup(XenPCIPassthroughState *s,

                          uint64_t addr,

                          uint32_t data,

                          int *ppirq,

                          bool is_msix,

                          int msix_entry,

                          bool is_not_mapped)

{

    uint8_t gvec = msi_vector(data);

    int rc = 0;



    assert((!is_msix && msix_entry == 0) || is_msix);



    if (gvec == 0) {

        /* if gvec is 0, the guest is asking for a particular pirq that

         * is passed as dest_id */

        *ppirq = msi_ext_dest_id(addr >> 32) | msi_dest_id(addr);

        if (!*ppirq) {

            /* this probably identifies an misconfiguration of the guest,

             * try the emulated path */

            *ppirq = XEN_PT_UNASSIGNED_PIRQ;

        } else {

            XEN_PT_LOG(&s->dev, "requested pirq %d for MSI%s"

                       " (vec: %#x, entry: %#x)\n",

                       *ppirq, is_msix ? "-X" : "", gvec, msix_entry);

        }

    }



    if (is_not_mapped) {

        uint64_t table_base = 0;



        if (is_msix) {

            table_base = s->msix->table_base;

        }



        rc = xc_physdev_map_pirq_msi(xen_xc, xen_domid, XEN_PT_AUTO_ASSIGN,

                                     ppirq, PCI_DEVFN(s->real_device.dev,

                                                      s->real_device.func),

                                     s->real_device.bus,

                                     msix_entry, table_base);

        if (rc) {

            XEN_PT_ERR(&s->dev,

                       "Mapping of MSI%s (err: %i, vec: %#x, entry %#x)\n",

                       is_msix ? "-X" : "", errno, gvec, msix_entry);

            return rc;

        }

    }



    return 0;

}
