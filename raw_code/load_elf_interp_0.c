static abi_ulong load_elf_interp(struct elfhdr * interp_elf_ex,

                                 int interpreter_fd,

                                 abi_ulong *interp_load_addr,

                                 char bprm_buf[BPRM_BUF_SIZE])

{

    struct elf_phdr *elf_phdata  =  NULL;

    abi_ulong load_addr, load_bias, loaddr, hiaddr;

    int retval;

    abi_ulong error;

    int i;



    bswap_ehdr(interp_elf_ex);

    /* First of all, some simple consistency checks */

    if ((interp_elf_ex->e_type != ET_EXEC &&

         interp_elf_ex->e_type != ET_DYN) ||

        !elf_check_arch(interp_elf_ex->e_machine)) {

        return ~((abi_ulong)0UL);

    }



    /* Now read in all of the header information */



    if (sizeof(struct elf_phdr) * interp_elf_ex->e_phnum > TARGET_PAGE_SIZE)

        return ~(abi_ulong)0UL;



    elf_phdata =  (struct elf_phdr *)

        malloc(sizeof(struct elf_phdr) * interp_elf_ex->e_phnum);



    if (!elf_phdata)

        return ~((abi_ulong)0UL);



    /*

     * If the size of this structure has changed, then punt, since

     * we will be doing the wrong thing.

     */

    if (interp_elf_ex->e_phentsize != sizeof(struct elf_phdr)) {

        free(elf_phdata);

        return ~((abi_ulong)0UL);

    }



    i = interp_elf_ex->e_phnum * sizeof(struct elf_phdr);

    if (interp_elf_ex->e_phoff + i <= BPRM_BUF_SIZE) {

        memcpy(elf_phdata, bprm_buf + interp_elf_ex->e_phoff, i);

    } else {

        retval = pread(interpreter_fd, elf_phdata, i, interp_elf_ex->e_phoff);

        if (retval != i) {

            perror("load_elf_interp");

            exit(-1);

        }

    }

    bswap_phdr(elf_phdata, interp_elf_ex->e_phnum);



    /* Find the maximum size of the image and allocate an appropriate

       amount of memory to handle that.  */

    loaddr = -1, hiaddr = 0;

    for (i = 0; i < interp_elf_ex->e_phnum; ++i) {

        if (elf_phdata[i].p_type == PT_LOAD) {

            abi_ulong a = elf_phdata[i].p_vaddr;

            if (a < loaddr) {

                loaddr = a;

            }

            a += elf_phdata[i].p_memsz;

            if (a > hiaddr) {

                hiaddr = a;

            }

        }

    }



    load_addr = loaddr;

    if (interp_elf_ex->e_type == ET_DYN) {

        /* The image indicates that it can be loaded anywhere.  Find a

           location that can hold the memory space required.  If the

           image is pre-linked, LOADDR will be non-zero.  Since we do

           not supply MAP_FIXED here we'll use that address if and

           only if it remains available.  */

        load_addr = target_mmap(loaddr, hiaddr - loaddr, PROT_NONE,

                                MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,

                                -1, 0);

        if (load_addr == -1) {

            perror("mmap");

            exit(-1);

        }

    }

    load_bias = load_addr - loaddr;



    for (i = 0; i < interp_elf_ex->e_phnum; i++) {

        struct elf_phdr *eppnt = elf_phdata + i;

        if (eppnt->p_type == PT_LOAD) {

            abi_ulong vaddr, vaddr_po, vaddr_ps, vaddr_ef, vaddr_em;

            int elf_prot = 0;



            if (eppnt->p_flags & PF_R) elf_prot =  PROT_READ;

            if (eppnt->p_flags & PF_W) elf_prot |= PROT_WRITE;

            if (eppnt->p_flags & PF_X) elf_prot |= PROT_EXEC;



            vaddr = load_bias + eppnt->p_vaddr;

            vaddr_po = TARGET_ELF_PAGEOFFSET(vaddr);

            vaddr_ps = TARGET_ELF_PAGESTART(vaddr);



            error = target_mmap(vaddr_ps, eppnt->p_filesz + vaddr_po,

                                elf_prot, MAP_PRIVATE | MAP_FIXED,

                                interpreter_fd, eppnt->p_offset - vaddr_po);

            if (error == -1) {

                /* Real error */

                close(interpreter_fd);

                free(elf_phdata);

                return ~((abi_ulong)0UL);

            }



            vaddr_ef = vaddr + eppnt->p_filesz;

            vaddr_em = vaddr + eppnt->p_memsz;



            /* If the load segment requests extra zeros (e.g. bss), map it.  */

            if (vaddr_ef < vaddr_em) {

                zero_bss(vaddr_ef, vaddr_em, elf_prot);

            }

        }

    }



    if (qemu_log_enabled()) {

        load_symbols(interp_elf_ex, interpreter_fd, load_bias);

    }



    close(interpreter_fd);

    free(elf_phdata);



    *interp_load_addr = load_addr;

    return ((abi_ulong) interp_elf_ex->e_entry) + load_bias;

}
