static abi_ulong mmap_find_vma_reserved(abi_ulong start, abi_ulong size)

{

    abi_ulong addr;

    abi_ulong end_addr;

    int prot;

    int looped = 0;



    if (size > reserved_va) {

        return (abi_ulong)-1;

    }



    size = HOST_PAGE_ALIGN(size);

    end_addr = start + size;

    if (end_addr > reserved_va) {

        end_addr = reserved_va;

    }

    addr = end_addr - qemu_host_page_size;



    while (1) {

        if (addr > end_addr) {

            if (looped) {

                return (abi_ulong)-1;

            }

            end_addr = reserved_va;

            addr = end_addr - qemu_host_page_size;

            looped = 1;

            continue;

        }

        prot = page_get_flags(addr);

        if (prot) {

            end_addr = addr;

        }

        if (addr + size == end_addr) {

            break;

        }

        addr -= qemu_host_page_size;

    }



    if (start == mmap_next_start) {

        mmap_next_start = addr;

    }



    return addr;

}
