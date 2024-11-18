void pci_register_vga(PCIDevice *pci_dev, MemoryRegion *mem,

                      MemoryRegion *io_lo, MemoryRegion *io_hi)

{

    assert(!pci_dev->has_vga);



    assert(memory_region_size(mem) == QEMU_PCI_VGA_MEM_SIZE);

    pci_dev->vga_regions[QEMU_PCI_VGA_MEM] = mem;

    memory_region_add_subregion_overlap(pci_dev->bus->address_space_mem,

                                        QEMU_PCI_VGA_MEM_BASE, mem, 1);



    assert(memory_region_size(io_lo) == QEMU_PCI_VGA_IO_LO_SIZE);

    pci_dev->vga_regions[QEMU_PCI_VGA_IO_LO] = io_lo;

    memory_region_add_subregion_overlap(pci_dev->bus->address_space_io,

                                        QEMU_PCI_VGA_IO_LO_BASE, io_lo, 1);



    assert(memory_region_size(io_hi) == QEMU_PCI_VGA_IO_HI_SIZE);

    pci_dev->vga_regions[QEMU_PCI_VGA_IO_HI] = io_hi;

    memory_region_add_subregion_overlap(pci_dev->bus->address_space_io,

                                        QEMU_PCI_VGA_IO_HI_BASE, io_hi, 1);

    pci_dev->has_vga = true;



    pci_update_vga(pci_dev);

}
