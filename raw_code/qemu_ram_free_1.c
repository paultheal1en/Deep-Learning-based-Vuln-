void qemu_ram_free(ram_addr_t addr)

{

    RAMBlock *block;



    QLIST_FOREACH(block, &ram_list.blocks, next) {

        if (addr == block->offset) {

            QLIST_REMOVE(block, next);

            if (block->flags & RAM_PREALLOC_MASK) {

                ;

            } else if (mem_path) {

#if defined (__linux__) && !defined(TARGET_S390X)

                if (block->fd) {

                    munmap(block->host, block->length);

                    close(block->fd);

                } else {

                    qemu_vfree(block->host);

                }



#endif

            } else {

#if defined(TARGET_S390X) && defined(CONFIG_KVM)

                munmap(block->host, block->length);


                qemu_vfree(block->host);

#endif

            }

            qemu_free(block);

            return;

        }

    }



}