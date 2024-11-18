static void pci_update_mappings(PCIDevice *d)

{

    PCIIORegion *r;

    int cmd, i;

    uint32_t last_addr, new_addr, config_ofs;



    cmd = le16_to_cpu(*(uint16_t *)(d->config + PCI_COMMAND));

    for(i = 0; i < PCI_NUM_REGIONS; i++) {

        r = &d->io_regions[i];

        if (i == PCI_ROM_SLOT) {

            config_ofs = 0x30;

        } else {

            config_ofs = 0x10 + i * 4;

        }

        if (r->size != 0) {

            if (r->type & PCI_ADDRESS_SPACE_IO) {

                if (cmd & PCI_COMMAND_IO) {

                    new_addr = le32_to_cpu(*(uint32_t *)(d->config +

                                                         config_ofs));

                    new_addr = new_addr & ~(r->size - 1);

                    last_addr = new_addr + r->size - 1;

                    /* NOTE: we have only 64K ioports on PC */

                    if (last_addr <= new_addr || new_addr == 0 ||

                        last_addr >= 0x10000) {

                        new_addr = -1;

                    }

                } else {

                    new_addr = -1;

                }

            } else {

                if (cmd & PCI_COMMAND_MEMORY) {

                    new_addr = le32_to_cpu(*(uint32_t *)(d->config +

                                                         config_ofs));

                    /* the ROM slot has a specific enable bit */

                    if (i == PCI_ROM_SLOT && !(new_addr & 1))

                        goto no_mem_map;

                    new_addr = new_addr & ~(r->size - 1);

                    last_addr = new_addr + r->size - 1;

                    /* NOTE: we do not support wrapping */

                    /* XXX: as we cannot support really dynamic

                       mappings, we handle specific values as invalid

                       mappings. */

                    if (last_addr <= new_addr || new_addr == 0 ||

                        last_addr == -1) {

                        new_addr = -1;

                    }

                } else {

                no_mem_map:

                    new_addr = -1;

                }

            }

            /* now do the real mapping */

            if (new_addr != r->addr) {

                if (r->addr != -1) {

                    if (r->type & PCI_ADDRESS_SPACE_IO) {

                        int class;

                        /* NOTE: specific hack for IDE in PC case:

                           only one byte must be mapped. */

                        class = d->config[0x0a] | (d->config[0x0b] << 8);

                        if (class == 0x0101 && r->size == 4) {

                            isa_unassign_ioport(r->addr + 2, 1);

                        } else {

                            isa_unassign_ioport(r->addr, r->size);

                        }

                    } else {

                        cpu_register_physical_memory(pci_to_cpu_addr(r->addr),

                                                     r->size,

                                                     IO_MEM_UNASSIGNED);


                    }

                }

                r->addr = new_addr;

                if (r->addr != -1) {

                    r->map_func(d, i, r->addr, r->size, r->type);

                }

            }

        }

    }

}