ram_addr_t qemu_ram_alloc_from_ptr(DeviceState *dev, const char *name,

                                   ram_addr_t size, void *host)

{

    RAMBlock *new_block, *block;



    size = TARGET_PAGE_ALIGN(size);

    new_block = qemu_mallocz(sizeof(*new_block));



    if (dev && dev->parent_bus && dev->parent_bus->info->get_dev_path) {

        char *id = dev->parent_bus->info->get_dev_path(dev);

        if (id) {

            snprintf(new_block->idstr, sizeof(new_block->idstr), "%s/", id);

            qemu_free(id);



    pstrcat(new_block->idstr, sizeof(new_block->idstr), name);



    QLIST_FOREACH(block, &ram_list.blocks, next) {

        if (!strcmp(block->idstr, new_block->idstr)) {

            fprintf(stderr, "RAMBlock \"%s\" already registered, abort!\n",

                    new_block->idstr);






    new_block->offset = find_ram_offset(size);

    if (host) {

        new_block->host = host;

        new_block->flags |= RAM_PREALLOC_MASK;

    } else {

        if (mem_path) {

#if defined (__linux__) && !defined(TARGET_S390X)

            new_block->host = file_ram_alloc(new_block, size, mem_path);

            if (!new_block->host) {

                new_block->host = qemu_vmalloc(size);

                qemu_madvise(new_block->host, size, QEMU_MADV_MERGEABLE);


#else

            fprintf(stderr, "-mem-path option unsupported\n");

            exit(1);

#endif

        } else {

#if defined(TARGET_S390X) && defined(CONFIG_KVM)

            /* S390 KVM requires the topmost vma of the RAM to be smaller than

               an system defined value, which is at least 256GB. Larger systems

               have larger values. We put the guest between the end of data

               segment (system break) and this value. We use 32GB as a base to

               have enough room for the system break to grow. */

            new_block->host = mmap((void*)0x800000000, size,

                                   PROT_EXEC|PROT_READ|PROT_WRITE,

                                   MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED, -1, 0);





#else

            if (xen_mapcache_enabled()) {

                xen_ram_alloc(new_block->offset, size);

            } else {

                new_block->host = qemu_vmalloc(size);


#endif

            qemu_madvise(new_block->host, size, QEMU_MADV_MERGEABLE);



    new_block->length = size;



    QLIST_INSERT_HEAD(&ram_list.blocks, new_block, next);



    ram_list.phys_dirty = qemu_realloc(ram_list.phys_dirty,

                                       last_ram_offset() >> TARGET_PAGE_BITS);

    memset(ram_list.phys_dirty + (new_block->offset >> TARGET_PAGE_BITS),

           0xff, size >> TARGET_PAGE_BITS);



    if (kvm_enabled())

        kvm_setup_guest_memory(new_block->host, size);



    return new_block->offset;
