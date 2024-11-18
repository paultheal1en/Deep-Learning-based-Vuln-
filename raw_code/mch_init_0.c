static int mch_init(PCIDevice *d)

{

    int i;

    MCHPCIState *mch = MCH_PCI_DEVICE(d);



    /* setup pci memory regions */

    memory_region_init_alias(&mch->pci_hole, OBJECT(mch), "pci-hole",

                             mch->pci_address_space,

                             mch->below_4g_mem_size,

                             0x100000000ULL - mch->below_4g_mem_size);

    memory_region_add_subregion(mch->system_memory, mch->below_4g_mem_size,

                                &mch->pci_hole);



    pc_init_pci64_hole(&mch->pci_info, 0x100000000ULL + mch->above_4g_mem_size,

                       mch->pci_hole64_size);

    memory_region_init_alias(&mch->pci_hole_64bit, OBJECT(mch), "pci-hole64",

                             mch->pci_address_space,

                             mch->pci_info.w64.begin,

                             mch->pci_hole64_size);

    if (mch->pci_hole64_size) {

        memory_region_add_subregion(mch->system_memory,

                                    mch->pci_info.w64.begin,

                                    &mch->pci_hole_64bit);

    }

    /* smram */

    cpu_smm_register(&mch_set_smm, mch);

    memory_region_init_alias(&mch->smram_region, OBJECT(mch), "smram-region",

                             mch->pci_address_space, 0xa0000, 0x20000);

    memory_region_add_subregion_overlap(mch->system_memory, 0xa0000,

                                        &mch->smram_region, 1);

    memory_region_set_enabled(&mch->smram_region, false);

    init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory, mch->pci_address_space,

             &mch->pam_regions[0], PAM_BIOS_BASE, PAM_BIOS_SIZE);

    for (i = 0; i < 12; ++i) {

        init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory, mch->pci_address_space,

                 &mch->pam_regions[i+1], PAM_EXPAN_BASE + i * PAM_EXPAN_SIZE,

                 PAM_EXPAN_SIZE);

    }

    return 0;

}
