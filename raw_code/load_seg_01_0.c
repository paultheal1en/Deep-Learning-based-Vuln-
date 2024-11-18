void load_seg(int seg_reg, int selector)

{

    SegmentCache *sc;

    SegmentDescriptorTable *dt;

    int index;

    uint32_t e1, e2;

    uint8_t *ptr;



    env->segs[seg_reg] = selector;

    sc = &env->seg_cache[seg_reg];

    if (env->eflags & VM_MASK) {

        sc->base = (void *)(selector << 4);

        sc->limit = 0xffff;

        sc->seg_32bit = 0;

    } else {

        if (selector & 0x4)

            dt = &env->ldt;

        else

            dt = &env->gdt;

        index = selector & ~7;

        if ((index + 7) > dt->limit)

            raise_exception_err(EXCP0D_GPF, selector);

        ptr = dt->base + index;

        e1 = ldl(ptr);

        e2 = ldl(ptr + 4);

        sc->base = (void *)((e1 >> 16) | ((e2 & 0xff) << 16) | (e2 & 0xff000000));

        sc->limit = (e1 & 0xffff) | (e2 & 0x000f0000);

        if (e2 & (1 << 23))

            sc->limit = (sc->limit << 12) | 0xfff;

        sc->seg_32bit = (e2 >> 22) & 1;

#if 0

        fprintf(logfile, "load_seg: sel=0x%04x base=0x%08lx limit=0x%08lx seg_32bit=%d\n", 

                selector, (unsigned long)sc->base, sc->limit, sc->seg_32bit);

#endif

    }

}
