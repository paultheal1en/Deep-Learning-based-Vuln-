static void vtd_iotlb_page_invalidate_notify(IntelIOMMUState *s,

                                           uint16_t domain_id, hwaddr addr,

                                           uint8_t am)

{

    IntelIOMMUNotifierNode *node;

    VTDContextEntry ce;

    int ret;



    QLIST_FOREACH(node, &(s->notifiers_list), next) {

        VTDAddressSpace *vtd_as = node->vtd_as;

        ret = vtd_dev_to_context_entry(s, pci_bus_num(vtd_as->bus),

                                       vtd_as->devfn, &ce);

        if (!ret && domain_id == VTD_CONTEXT_ENTRY_DID(ce.hi)) {

            vtd_page_walk(&ce, addr, addr + (1 << am) * VTD_PAGE_SIZE,

                          vtd_page_invalidate_notify_hook,

                          (void *)&vtd_as->iommu, true);

        }

    }

}
