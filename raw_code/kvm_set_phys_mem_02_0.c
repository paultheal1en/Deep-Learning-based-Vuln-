static void kvm_set_phys_mem(MemoryRegionSection *section, bool add)

{

    KVMState *s = kvm_state;

    KVMSlot *mem, old;

    int err;

    MemoryRegion *mr = section->mr;

    bool log_dirty = memory_region_is_logging(mr);

    bool writeable = !mr->readonly && !mr->rom_device;

    bool readonly_flag = mr->readonly || memory_region_is_romd(mr);

    hwaddr start_addr = section->offset_within_address_space;

    ram_addr_t size = int128_get64(section->size);

    void *ram = NULL;

    unsigned delta;



    /* kvm works in page size chunks, but the function may be called

       with sub-page size and unaligned start address. Pad the start

       address to next and truncate size to previous page boundary. */

    delta = (TARGET_PAGE_SIZE - (start_addr & ~TARGET_PAGE_MASK));

    delta &= ~TARGET_PAGE_MASK;

    if (delta > size) {

        return;

    }

    start_addr += delta;

    size -= delta;

    size &= TARGET_PAGE_MASK;

    if (!size || (start_addr & ~TARGET_PAGE_MASK)) {

        return;

    }



    if (!memory_region_is_ram(mr)) {

        if (writeable || !kvm_readonly_mem_allowed) {

            return;

        } else if (!mr->romd_mode) {

            /* If the memory device is not in romd_mode, then we actually want

             * to remove the kvm memory slot so all accesses will trap. */

            add = false;

        }

    }



    ram = memory_region_get_ram_ptr(mr) + section->offset_within_region + delta;



    while (1) {

        mem = kvm_lookup_overlapping_slot(s, start_addr, start_addr + size);

        if (!mem) {

            break;

        }



        if (add && start_addr >= mem->start_addr &&

            (start_addr + size <= mem->start_addr + mem->memory_size) &&

            (ram - start_addr == mem->ram - mem->start_addr)) {

            /* The new slot fits into the existing one and comes with

             * identical parameters - update flags and done. */

            kvm_slot_dirty_pages_log_change(mem, log_dirty);

            return;

        }



        old = *mem;



        if ((mem->flags & KVM_MEM_LOG_DIRTY_PAGES) || s->migration_log) {

            kvm_physical_sync_dirty_bitmap(section);

        }



        /* unregister the overlapping slot */

        mem->memory_size = 0;

        err = kvm_set_user_memory_region(s, mem);

        if (err) {

            fprintf(stderr, "%s: error unregistering overlapping slot: %s\n",

                    __func__, strerror(-err));

            abort();

        }



        /* Workaround for older KVM versions: we can't join slots, even not by

         * unregistering the previous ones and then registering the larger

         * slot. We have to maintain the existing fragmentation. Sigh.

         *

         * This workaround assumes that the new slot starts at the same

         * address as the first existing one. If not or if some overlapping

         * slot comes around later, we will fail (not seen in practice so far)

         * - and actually require a recent KVM version. */

        if (s->broken_set_mem_region &&

            old.start_addr == start_addr && old.memory_size < size && add) {

            mem = kvm_alloc_slot(s);

            mem->memory_size = old.memory_size;

            mem->start_addr = old.start_addr;

            mem->ram = old.ram;

            mem->flags = kvm_mem_flags(s, log_dirty, readonly_flag);



            err = kvm_set_user_memory_region(s, mem);

            if (err) {

                fprintf(stderr, "%s: error updating slot: %s\n", __func__,

                        strerror(-err));

                abort();

            }



            start_addr += old.memory_size;

            ram += old.memory_size;

            size -= old.memory_size;

            continue;

        }



        /* register prefix slot */

        if (old.start_addr < start_addr) {

            mem = kvm_alloc_slot(s);

            mem->memory_size = start_addr - old.start_addr;

            mem->start_addr = old.start_addr;

            mem->ram = old.ram;

            mem->flags =  kvm_mem_flags(s, log_dirty, readonly_flag);



            err = kvm_set_user_memory_region(s, mem);

            if (err) {

                fprintf(stderr, "%s: error registering prefix slot: %s\n",

                        __func__, strerror(-err));

#ifdef TARGET_PPC

                fprintf(stderr, "%s: This is probably because your kernel's " \

                                "PAGE_SIZE is too big. Please try to use 4k " \

                                "PAGE_SIZE!\n", __func__);

#endif

                abort();

            }

        }



        /* register suffix slot */

        if (old.start_addr + old.memory_size > start_addr + size) {

            ram_addr_t size_delta;



            mem = kvm_alloc_slot(s);

            mem->start_addr = start_addr + size;

            size_delta = mem->start_addr - old.start_addr;

            mem->memory_size = old.memory_size - size_delta;

            mem->ram = old.ram + size_delta;

            mem->flags = kvm_mem_flags(s, log_dirty, readonly_flag);



            err = kvm_set_user_memory_region(s, mem);

            if (err) {

                fprintf(stderr, "%s: error registering suffix slot: %s\n",

                        __func__, strerror(-err));

                abort();

            }

        }

    }



    /* in case the KVM bug workaround already "consumed" the new slot */

    if (!size) {

        return;

    }

    if (!add) {

        return;

    }

    mem = kvm_alloc_slot(s);

    mem->memory_size = size;

    mem->start_addr = start_addr;

    mem->ram = ram;

    mem->flags = kvm_mem_flags(s, log_dirty, readonly_flag);



    err = kvm_set_user_memory_region(s, mem);

    if (err) {

        fprintf(stderr, "%s: error registering slot: %s\n", __func__,

                strerror(-err));

        abort();

    }

}
