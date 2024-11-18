static int elf_core_dump(int signr, const CPUArchState *env)

{

    const CPUState *cpu = ENV_GET_CPU((CPUArchState *)env);

    const TaskState *ts = (const TaskState *)cpu->opaque;

    struct vm_area_struct *vma = NULL;

    char corefile[PATH_MAX];

    struct elf_note_info info;

    struct elfhdr elf;

    struct elf_phdr phdr;

    struct rlimit dumpsize;

    struct mm_struct *mm = NULL;

    off_t offset = 0, data_offset = 0;

    int segs = 0;

    int fd = -1;



    init_note_info(&info);



    errno = 0;

    getrlimit(RLIMIT_CORE, &dumpsize);

    if (dumpsize.rlim_cur == 0)

        return 0;



    if (core_dump_filename(ts, corefile, sizeof (corefile)) < 0)

        return (-errno);



    if ((fd = open(corefile, O_WRONLY | O_CREAT,

                   S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)) < 0)

        return (-errno);



    /*

     * Walk through target process memory mappings and

     * set up structure containing this information.  After

     * this point vma_xxx functions can be used.

     */

    if ((mm = vma_init()) == NULL)

        goto out;



    walk_memory_regions(mm, vma_walker);

    segs = vma_get_mapping_count(mm);



    /*

     * Construct valid coredump ELF header.  We also

     * add one more segment for notes.

     */

    fill_elf_header(&elf, segs + 1, ELF_MACHINE, 0);

    if (dump_write(fd, &elf, sizeof (elf)) != 0)

        goto out;



    /* fill in the in-memory version of notes */

    if (fill_note_info(&info, signr, env) < 0)

        goto out;



    offset += sizeof (elf);                             /* elf header */

    offset += (segs + 1) * sizeof (struct elf_phdr);    /* program headers */



    /* write out notes program header */

    fill_elf_note_phdr(&phdr, info.notes_size, offset);



    offset += info.notes_size;

    if (dump_write(fd, &phdr, sizeof (phdr)) != 0)

        goto out;



    /*

     * ELF specification wants data to start at page boundary so

     * we align it here.

     */

    data_offset = offset = roundup(offset, ELF_EXEC_PAGESIZE);



    /*

     * Write program headers for memory regions mapped in

     * the target process.

     */

    for (vma = vma_first(mm); vma != NULL; vma = vma_next(vma)) {

        (void) memset(&phdr, 0, sizeof (phdr));



        phdr.p_type = PT_LOAD;

        phdr.p_offset = offset;

        phdr.p_vaddr = vma->vma_start;

        phdr.p_paddr = 0;

        phdr.p_filesz = vma_dump_size(vma);

        offset += phdr.p_filesz;

        phdr.p_memsz = vma->vma_end - vma->vma_start;

        phdr.p_flags = vma->vma_flags & PROT_READ ? PF_R : 0;

        if (vma->vma_flags & PROT_WRITE)

            phdr.p_flags |= PF_W;

        if (vma->vma_flags & PROT_EXEC)

            phdr.p_flags |= PF_X;

        phdr.p_align = ELF_EXEC_PAGESIZE;



        bswap_phdr(&phdr, 1);

        dump_write(fd, &phdr, sizeof (phdr));

    }



    /*

     * Next we write notes just after program headers.  No

     * alignment needed here.

     */

    if (write_note_info(&info, fd) < 0)

        goto out;



    /* align data to page boundary */

    if (lseek(fd, data_offset, SEEK_SET) != data_offset)

        goto out;



    /*

     * Finally we can dump process memory into corefile as well.

     */

    for (vma = vma_first(mm); vma != NULL; vma = vma_next(vma)) {

        abi_ulong addr;

        abi_ulong end;



        end = vma->vma_start + vma_dump_size(vma);



        for (addr = vma->vma_start; addr < end;

             addr += TARGET_PAGE_SIZE) {

            char page[TARGET_PAGE_SIZE];

            int error;



            /*

             *  Read in page from target process memory and

             *  write it to coredump file.

             */

            error = copy_from_user(page, addr, sizeof (page));

            if (error != 0) {

                (void) fprintf(stderr, "unable to dump " TARGET_ABI_FMT_lx "\n",

                               addr);

                errno = -error;

                goto out;

            }

            if (dump_write(fd, page, TARGET_PAGE_SIZE) < 0)

                goto out;

        }

    }



 out:

    free_note_info(&info);

    if (mm != NULL)

        vma_delete(mm);

    (void) close(fd);



    if (errno != 0)

        return (-errno);

    return (0);

}
