void kvm_set_phys_mem(target_phys_addr_t start_addr,

                      ram_addr_t size,

                      ram_addr_t phys_offset)

{

    KVMState *s = kvm_state;

    ram_addr_t flags = phys_offset & ~TARGET_PAGE_MASK;

    KVMSlot *mem;



    if (start_addr & ~TARGET_PAGE_MASK) {

        fprintf(stderr, "Only page-aligned memory slots supported\n");

        abort();

    }



    /* KVM does not support read-only slots */

    phys_offset &= ~IO_MEM_ROM;



    mem = kvm_lookup_slot(s, start_addr);

    if (mem) {

        if (flags >= IO_MEM_UNASSIGNED) {

            mem->memory_size = 0;

            mem->start_addr = start_addr;

            mem->phys_offset = 0;

            mem->flags = 0;



            kvm_set_user_memory_region(s, mem);

        } else if (start_addr >= mem->start_addr &&

                   (start_addr + size) <= (mem->start_addr +

                                           mem->memory_size)) {

            KVMSlot slot;

            target_phys_addr_t mem_start;

            ram_addr_t mem_size, mem_offset;



            /* Not splitting */

            if ((phys_offset - (start_addr - mem->start_addr)) == 

                mem->phys_offset)

                return;



            /* unregister whole slot */

            memcpy(&slot, mem, sizeof(slot));

            mem->memory_size = 0;

            kvm_set_user_memory_region(s, mem);



            /* register prefix slot */

            mem_start = slot.start_addr;

            mem_size = start_addr - slot.start_addr;

            mem_offset = slot.phys_offset;

            if (mem_size)

                kvm_set_phys_mem(mem_start, mem_size, mem_offset);



            /* register new slot */

            kvm_set_phys_mem(start_addr, size, phys_offset);



            /* register suffix slot */

            mem_start = start_addr + size;

            mem_offset += mem_size + size;

            mem_size = slot.memory_size - mem_size - size;

            if (mem_size)

                kvm_set_phys_mem(mem_start, mem_size, mem_offset);



            return;

        } else {

            printf("Registering overlapping slot\n");

            abort();

        }

    }

    /* KVM does not need to know about this memory */

    if (flags >= IO_MEM_UNASSIGNED)

        return;



    mem = kvm_alloc_slot(s);

    mem->memory_size = size;

    mem->start_addr = start_addr;

    mem->phys_offset = phys_offset;

    mem->flags = 0;



    kvm_set_user_memory_region(s, mem);

    /* FIXME deal with errors */

}
