static void qxl_add_memslot(PCIQXLDevice *d, uint32_t slot_id, uint64_t delta,

                            qxl_async_io async)

{

    static const int regions[] = {

        QXL_RAM_RANGE_INDEX,

        QXL_VRAM_RANGE_INDEX,

        QXL_VRAM64_RANGE_INDEX,

    };

    uint64_t guest_start;

    uint64_t guest_end;

    int pci_region;

    pcibus_t pci_start;

    pcibus_t pci_end;

    intptr_t virt_start;

    QXLDevMemSlot memslot;

    int i;



    guest_start = le64_to_cpu(d->guest_slots[slot_id].slot.mem_start);

    guest_end   = le64_to_cpu(d->guest_slots[slot_id].slot.mem_end);



    trace_qxl_memslot_add_guest(d->id, slot_id, guest_start, guest_end);



    PANIC_ON(slot_id >= NUM_MEMSLOTS);

    PANIC_ON(guest_start > guest_end);



    for (i = 0; i < ARRAY_SIZE(regions); i++) {

        pci_region = regions[i];

        pci_start = d->pci.io_regions[pci_region].addr;

        pci_end = pci_start + d->pci.io_regions[pci_region].size;

        /* mapped? */

        if (pci_start == -1) {

            continue;

        }

        /* start address in range ? */

        if (guest_start < pci_start || guest_start > pci_end) {

            continue;

        }

        /* end address in range ? */

        if (guest_end > pci_end) {

            continue;

        }

        /* passed */

        break;

    }

    PANIC_ON(i == ARRAY_SIZE(regions)); /* finished loop without match */



    switch (pci_region) {

    case QXL_RAM_RANGE_INDEX:

        virt_start = (intptr_t)memory_region_get_ram_ptr(&d->vga.vram);

        break;

    case QXL_VRAM_RANGE_INDEX:

    case 4 /* vram 64bit */:

        virt_start = (intptr_t)memory_region_get_ram_ptr(&d->vram_bar);

        break;

    default:

        /* should not happen */

        abort();

    }



    memslot.slot_id = slot_id;

    memslot.slot_group_id = MEMSLOT_GROUP_GUEST; /* guest group */

    memslot.virt_start = virt_start + (guest_start - pci_start);

    memslot.virt_end   = virt_start + (guest_end   - pci_start);

    memslot.addr_delta = memslot.virt_start - delta;

    memslot.generation = d->rom->slot_generation = 0;

    qxl_rom_set_dirty(d);



    qemu_spice_add_memslot(&d->ssd, &memslot, async);

    d->guest_slots[slot_id].ptr = (void*)memslot.virt_start;

    d->guest_slots[slot_id].size = memslot.virt_end - memslot.virt_start;

    d->guest_slots[slot_id].delta = delta;

    d->guest_slots[slot_id].active = 1;

}
