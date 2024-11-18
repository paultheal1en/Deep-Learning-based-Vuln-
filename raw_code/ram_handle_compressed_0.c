void ram_handle_compressed(void *host, uint8_t ch, uint64_t size)

{

    if (ch != 0 || !is_zero_range(host, size)) {

        memset(host, ch, size);

#ifndef _WIN32

        if (ch == 0 && (!kvm_enabled() || kvm_has_sync_mmu())) {

            size = size & ~(getpagesize() - 1);

            if (size > 0) {

                qemu_madvise(host, size, QEMU_MADV_DONTNEED);

            }

        }

#endif

    }

}
