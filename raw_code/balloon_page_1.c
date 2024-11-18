static void balloon_page(void *addr, int deflate)

{

#if defined(__linux__)

    if (!kvm_enabled() || kvm_has_sync_mmu())

        madvise(addr, TARGET_PAGE_SIZE,

                deflate ? MADV_WILLNEED : MADV_DONTNEED);

#endif

}
