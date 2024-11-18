int ram_load(QEMUFile *f, void *opaque, int version_id)

{

    ram_addr_t addr;

    int flags;



    if (version_id < 3 || version_id > 4) {

        return -EINVAL;

    }



    do {

        addr = qemu_get_be64(f);



        flags = addr & ~TARGET_PAGE_MASK;

        addr &= TARGET_PAGE_MASK;



        if (flags & RAM_SAVE_FLAG_MEM_SIZE) {

            if (version_id == 3) {

                if (addr != ram_bytes_total()) {

                    return -EINVAL;

                }

            } else {

                /* Synchronize RAM block list */

                char id[256];

                ram_addr_t length;

                ram_addr_t total_ram_bytes = addr;



                while (total_ram_bytes) {

                    RAMBlock *block;

                    uint8_t len;



                    len = qemu_get_byte(f);

                    qemu_get_buffer(f, (uint8_t *)id, len);

                    id[len] = 0;

                    length = qemu_get_be64(f);



                    QLIST_FOREACH(block, &ram_list.blocks, next) {

                        if (!strncmp(id, block->idstr, sizeof(id))) {

                            if (block->length != length)

                                return -EINVAL;

                            break;

                        }

                    }



                    if (!block) {

                        fprintf(stderr, "Unknown ramblock \"%s\", cannot "

                                "accept migration\n", id);

                        return -EINVAL;

                    }



                    total_ram_bytes -= length;

                }

            }

        }



        if (flags & RAM_SAVE_FLAG_COMPRESS) {

            void *host;

            uint8_t ch;



            if (version_id == 3)

                host = qemu_get_ram_ptr(addr);

            else

                host = host_from_stream_offset(f, addr, flags);



            ch = qemu_get_byte(f);

            memset(host, ch, TARGET_PAGE_SIZE);

#ifndef _WIN32

            if (ch == 0 &&

                (!kvm_enabled() || kvm_has_sync_mmu())) {

                madvise(host, TARGET_PAGE_SIZE, MADV_DONTNEED);

            }

#endif

        } else if (flags & RAM_SAVE_FLAG_PAGE) {

            void *host;



            if (version_id == 3)

                host = qemu_get_ram_ptr(addr);

            else

                host = host_from_stream_offset(f, addr, flags);



            qemu_get_buffer(f, host, TARGET_PAGE_SIZE);

        }

        if (qemu_file_has_error(f)) {

            return -EIO;

        }

    } while (!(flags & RAM_SAVE_FLAG_EOS));



    return 0;

}
