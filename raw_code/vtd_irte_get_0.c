static int vtd_irte_get(IntelIOMMUState *iommu, uint16_t index,

                        VTD_IRTE *entry)

{

    dma_addr_t addr = 0x00;



    addr = iommu->intr_root + index * sizeof(*entry);

    if (dma_memory_read(&address_space_memory, addr, entry,

                        sizeof(*entry))) {

        VTD_DPRINTF(GENERAL, "error: fail to access IR root at 0x%"PRIx64

                    " + %"PRIu16, iommu->intr_root, index);

        return -VTD_FR_IR_ROOT_INVAL;

    }



    if (!entry->present) {

        VTD_DPRINTF(GENERAL, "error: present flag not set in IRTE"

                    " entry index %u value 0x%"PRIx64 " 0x%"PRIx64,

                    index, le64_to_cpu(entry->data[1]),

                    le64_to_cpu(entry->data[0]));

        return -VTD_FR_IR_ENTRY_P;

    }



    if (entry->__reserved_0 || entry->__reserved_1 || \

        entry->__reserved_2) {

        VTD_DPRINTF(GENERAL, "error: IRTE entry index %"PRIu16

                    " reserved fields non-zero: 0x%"PRIx64 " 0x%"PRIx64,

                    index, le64_to_cpu(entry->data[1]),

                    le64_to_cpu(entry->data[0]));

        return -VTD_FR_IR_IRTE_RSVD;

    }



    /*

     * TODO: Check Source-ID corresponds to SVT (Source Validation

     * Type) bits

     */



    return 0;

}
