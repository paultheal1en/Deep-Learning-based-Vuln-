ram_addr_t qemu_ram_alloc(ram_addr_t size)

{

    RAMBlock *new_block;



#ifdef CONFIG_KQEMU

    if (kqemu_phys_ram_base) {

        return kqemu_ram_alloc(size);

    }

#endif



    size = TARGET_PAGE_ALIGN(size);

    new_block = qemu_malloc(sizeof(*new_block));



    new_block->host = qemu_vmalloc(size);

    new_block->offset = last_ram_offset;

    new_block->length = size;



    new_block->next = ram_blocks;

    ram_blocks = new_block;



    phys_ram_dirty = qemu_realloc(phys_ram_dirty,

        (last_ram_offset + size) >> TARGET_PAGE_BITS);

    memset(phys_ram_dirty + (last_ram_offset >> TARGET_PAGE_BITS),

           0xff, size >> TARGET_PAGE_BITS);



    last_ram_offset += size;



    if (kvm_enabled())

        kvm_setup_guest_memory(new_block->host, size);



    return new_block->offset;

}
