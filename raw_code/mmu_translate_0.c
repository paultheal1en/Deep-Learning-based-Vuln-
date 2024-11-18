int mmu_translate(CPUS390XState *env, target_ulong vaddr, int rw, uint64_t asc,

                  target_ulong *raddr, int *flags)

{

    int r = -1;

    uint8_t *sk;



    *flags = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

    vaddr &= TARGET_PAGE_MASK;



    if (!(env->psw.mask & PSW_MASK_DAT)) {

        *raddr = vaddr;

        r = 0;

        goto out;

    }



    switch (asc) {

    case PSW_ASC_PRIMARY:

    case PSW_ASC_HOME:

        r = mmu_translate_asc(env, vaddr, asc, raddr, flags, rw);

        break;

    case PSW_ASC_SECONDARY:

        /*

         * Instruction: Primary

         * Data: Secondary

         */

        if (rw == 2) {

            r = mmu_translate_asc(env, vaddr, PSW_ASC_PRIMARY, raddr, flags,

                                  rw);

            *flags &= ~(PAGE_READ | PAGE_WRITE);

        } else {

            r = mmu_translate_asc(env, vaddr, PSW_ASC_SECONDARY, raddr, flags,

                                  rw);

            *flags &= ~(PAGE_EXEC);

        }

        break;

    case PSW_ASC_ACCREG:

    default:

        hw_error("guest switched to unknown asc mode\n");

        break;

    }



 out:

    /* Convert real address -> absolute address */

    *raddr = mmu_real2abs(env, *raddr);



    if (*raddr <= ram_size) {

        sk = &env->storage_keys[*raddr / TARGET_PAGE_SIZE];

        if (*flags & PAGE_READ) {

            *sk |= SK_R;

        }



        if (*flags & PAGE_WRITE) {

            *sk |= SK_C;

        }

    }



    return r;

}
