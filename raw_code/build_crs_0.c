static Aml *build_crs(PCIHostState *host,

                      GPtrArray *io_ranges, GPtrArray *mem_ranges)

{

    Aml *crs = aml_resource_template();

    uint8_t max_bus = pci_bus_num(host->bus);

    uint8_t type;

    int devfn;



    for (devfn = 0; devfn < ARRAY_SIZE(host->bus->devices); devfn++) {

        int i;

        uint64_t range_base, range_limit;

        PCIDevice *dev = host->bus->devices[devfn];



        if (!dev) {

            continue;

        }



        for (i = 0; i < PCI_NUM_REGIONS; i++) {

            PCIIORegion *r = &dev->io_regions[i];



            range_base = r->addr;

            range_limit = r->addr + r->size - 1;



            /*

             * Work-around for old bioses

             * that do not support multiple root buses

             */

            if (!range_base || range_base > range_limit) {

                continue;

            }



            if (r->type & PCI_BASE_ADDRESS_SPACE_IO) {

                aml_append(crs,

                    aml_word_io(AML_MIN_FIXED, AML_MAX_FIXED,

                                AML_POS_DECODE, AML_ENTIRE_RANGE,

                                0,

                                range_base,

                                range_limit,

                                0,

                                range_limit - range_base + 1));

                crs_range_insert(io_ranges, range_base, range_limit);

            } else { /* "memory" */

                aml_append(crs,

                    aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED,

                                     AML_MAX_FIXED, AML_NON_CACHEABLE,

                                     AML_READ_WRITE,

                                     0,

                                     range_base,

                                     range_limit,

                                     0,

                                     range_limit - range_base + 1));

                crs_range_insert(mem_ranges, range_base, range_limit);

            }

        }



        type = dev->config[PCI_HEADER_TYPE] & ~PCI_HEADER_TYPE_MULTI_FUNCTION;

        if (type == PCI_HEADER_TYPE_BRIDGE) {

            uint8_t subordinate = dev->config[PCI_SUBORDINATE_BUS];

            if (subordinate > max_bus) {

                max_bus = subordinate;

            }



            range_base = pci_bridge_get_base(dev, PCI_BASE_ADDRESS_SPACE_IO);

            range_limit = pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_SPACE_IO);



            /*

             * Work-around for old bioses

             * that do not support multiple root buses

             */

            if (range_base || range_base > range_limit) {

                aml_append(crs,

                           aml_word_io(AML_MIN_FIXED, AML_MAX_FIXED,

                                       AML_POS_DECODE, AML_ENTIRE_RANGE,

                                       0,

                                       range_base,

                                       range_limit,

                                       0,

                                       range_limit - range_base + 1));

                crs_range_insert(io_ranges, range_base, range_limit);

            }



            range_base =

                pci_bridge_get_base(dev, PCI_BASE_ADDRESS_SPACE_MEMORY);

            range_limit =

                pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_SPACE_MEMORY);



            /*

             * Work-around for old bioses

             * that do not support multiple root buses

             */

            if (range_base || range_base > range_limit) {

                aml_append(crs,

                           aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED,

                                            AML_MAX_FIXED, AML_NON_CACHEABLE,

                                            AML_READ_WRITE,

                                            0,

                                            range_base,

                                            range_limit,

                                            0,

                                            range_limit - range_base + 1));

                crs_range_insert(mem_ranges, range_base, range_limit);

          }



            range_base =

                pci_bridge_get_base(dev, PCI_BASE_ADDRESS_MEM_PREFETCH);

            range_limit =

                pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_MEM_PREFETCH);



            /*

             * Work-around for old bioses

             * that do not support multiple root buses

             */

            if (range_base || range_base > range_limit) {

                aml_append(crs,

                           aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED,

                                            AML_MAX_FIXED, AML_NON_CACHEABLE,

                                            AML_READ_WRITE,

                                            0,

                                            range_base,

                                            range_limit,

                                            0,

                                            range_limit - range_base + 1));

                crs_range_insert(mem_ranges, range_base, range_limit);

            }

        }

    }



    aml_append(crs,

        aml_word_bus_number(AML_MIN_FIXED, AML_MAX_FIXED, AML_POS_DECODE,

                            0,

                            pci_bus_num(host->bus),

                            max_bus,

                            0,

                            max_bus - pci_bus_num(host->bus) + 1));



    return crs;

}
