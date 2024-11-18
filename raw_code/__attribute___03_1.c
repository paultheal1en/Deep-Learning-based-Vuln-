static __attribute__((unused)) void map_exec(void *addr, long size)

{

    unsigned long start, end, page_size;



    page_size = getpagesize();

    start = (unsigned long)addr;

    start &= ~(page_size - 1);



    end = (unsigned long)addr + size;

    end += page_size - 1;

    end &= ~(page_size - 1);



    mprotect((void *)start, end - start,

             PROT_READ | PROT_WRITE | PROT_EXEC);

}
