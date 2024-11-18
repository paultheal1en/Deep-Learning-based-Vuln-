void *vnc_zlib_zalloc(void *x, unsigned items, unsigned size)

{

    void *p;



    size *= items;

    size = (size + ZALLOC_ALIGNMENT - 1) & ~(ZALLOC_ALIGNMENT - 1);



    p = qemu_mallocz(size);



    return (p);

}
