static int vma_add_mapping(struct mm_struct *mm, abi_ulong start,

    abi_ulong end, abi_ulong flags)

{

    struct vm_area_struct *vma;



    if ((vma = qemu_mallocz(sizeof (*vma))) == NULL)

        return (-1);



    vma->vma_start = start;

    vma->vma_end = end;

    vma->vma_flags = flags;



    TAILQ_INSERT_TAIL(&mm->mm_mmap, vma, vma_link);

    mm->mm_count++;



    return (0);

}
