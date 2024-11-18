int load_flt_binary(struct linux_binprm * bprm, struct target_pt_regs * regs,

                    struct image_info * info)

{

    struct lib_info libinfo[MAX_SHARED_LIBS];

    abi_ulong p = bprm->p;

    abi_ulong stack_len;

    abi_ulong start_addr;

    abi_ulong sp;

    int res;

    int i, j;



    memset(libinfo, 0, sizeof(libinfo));

    /*

     * We have to add the size of our arguments to our stack size

     * otherwise it's too easy for users to create stack overflows

     * by passing in a huge argument list.  And yes,  we have to be

     * pedantic and include space for the argv/envp array as it may have

     * a lot of entries.

     */

    stack_len = 0;

    for (i = 0; i < bprm->argc; ++i) {

        /* the argv strings */

        stack_len += strlen(bprm->argv[i]);

    }

    for (i = 0; i < bprm->envc; ++i) {

        /* the envp strings */

        stack_len += strlen(bprm->envp[i]);

    }

    stack_len += (bprm->argc + 1) * 4; /* the argv array */

    stack_len += (bprm->envc + 1) * 4; /* the envp array */





    res = load_flat_file(bprm, libinfo, 0, &stack_len);

    if (res > (unsigned long)-4096)

            return res;



    /* Update data segment pointers for all libraries */

    for (i=0; i<MAX_SHARED_LIBS; i++) {

        if (libinfo[i].loaded) {

            abi_ulong p;

            p = libinfo[i].start_data;

            for (j=0; j<MAX_SHARED_LIBS; j++) {

                p -= 4;

                /* FIXME - handle put_user() failures */

                if (put_user_ual(libinfo[j].loaded

                                 ? libinfo[j].start_data

                                 : UNLOADED_LIB,

                                 p))

                    return -EFAULT;

            }

        }

    }



    p = ((libinfo[0].start_brk + stack_len + 3) & ~3) - 4;

    DBG_FLT("p=%x\n", (int)p);



    /* Copy argv/envp.  */

    p = copy_strings(p, bprm->envc, bprm->envp);

    p = copy_strings(p, bprm->argc, bprm->argv);

    /* Align stack.  */

    sp = p & ~(abi_ulong)(sizeof(abi_ulong) - 1);

    /* Enforce final stack alignment of 16 bytes.  This is sufficient

       for all current targets, and excess alignment is harmless.  */

    stack_len = bprm->envc + bprm->argc + 2;

    stack_len += 3;	/* argc, arvg, argp */

    stack_len *= sizeof(abi_ulong);

    if ((sp + stack_len) & 15)

        sp -= 16 - ((sp + stack_len) & 15);

    sp = loader_build_argptr(bprm->envc, bprm->argc, sp, p, 1);



    /* Fake some return addresses to ensure the call chain will

     * initialise library in order for us.  We are required to call

     * lib 1 first, then 2, ... and finally the main program (id 0).

     */

    start_addr = libinfo[0].entry;



#ifdef CONFIG_BINFMT_SHARED_FLAT

#error here

    for (i = MAX_SHARED_LIBS-1; i>0; i--) {

            if (libinfo[i].loaded) {

                    /* Push previos first to call address */

                    --sp;

                    if (put_user_ual(start_addr, sp))

                        return -EFAULT;

                    start_addr = libinfo[i].entry;

            }

    }

#endif



    /* Stash our initial stack pointer into the mm structure */

    info->start_code = libinfo[0].start_code;

    info->end_code = libinfo[0].start_code = libinfo[0].text_len;

    info->start_data = libinfo[0].start_data;

    info->end_data = libinfo[0].end_data;

    info->start_brk = libinfo[0].start_brk;

    info->start_stack = sp;

    info->stack_limit = libinfo[0].start_brk;

    info->entry = start_addr;

    info->code_offset = info->start_code;

    info->data_offset = info->start_data - libinfo[0].text_len;



    DBG_FLT("start_thread(entry=0x%x, start_stack=0x%x)\n",

            (int)info->entry, (int)info->start_stack);



    return 0;

}
