void *qemu_memalign(size_t alignment, size_t size)

{

    return VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);

}
